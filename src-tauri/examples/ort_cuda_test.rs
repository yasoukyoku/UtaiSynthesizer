// Normal ort load path (init_from + commit + CUDA session) — now testing with a VERSION-MATCHED CUDA
// build (ORT 1.24.x, API 24, matching the DirectML build 1.24.4 and ort crate 2.0-rc.12). The earlier
// hangs were a 4-major-version API mismatch (CUDA build was 1.20.1 / API 20 vs ort wanting API 24).
// Run: cargo run --example ort_cuda_test   (from src-tauri)

use std::time::Duration;

fn main() {
    let cuda_dll = std::path::PathBuf::from(r"D:\MyDev\Utai_v2-dev\runtime\ort\cuda\onnxruntime.dll");
    let cuda_bin = std::path::PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin");
    let cudnn_dir = std::path::PathBuf::from(r"D:\MyDev\Utai_v2-dev\runtime\cuda");
    let model = r"C:\Users\admin\AppData\Local\com.utai.app\models\msst\model_bs_roformer_ep_317_sdr_12.9755.onnx";

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        eprintln!("[0] preload_dylibs ...");
        let _ = ort::ep::cuda::preload_dylibs(Some(&cuda_bin), Some(&cudnn_dir));
        eprintln!("[1] ort::init_from + commit ...");
        match ort::init_from(&cuda_dll) {
            Ok(b) => {
                let _ = b.commit();
                eprintln!("[2] ORT environment committed OK");
            }
            Err(e) => {
                eprintln!("[!] init_from failed: {}", e);
                let _ = tx.send(());
                return;
            }
        }
        eprintln!("[3] building CUDA session ...");
        let result = (|| -> Result<(), String> {
            // Replicate the main app's build_session_auto EXACTLY: Level3 optimization + error_on_failure.
            let builder = ort::session::Session::builder().map_err(|e| e.to_string())?
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3).map_err(|e| e.to_string())?;
            let mut builder = builder
                .with_execution_providers([ort::ep::CUDA::default()
                    .with_tf32(true)
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                    .with_conv_max_workspace(false)
                    .build()
                    .error_on_failure()])
                .map_err(|e| e.to_string())?;
            let mut session = builder.commit_from_file(model).map_err(|e| e.to_string())?;
            // REAL frames (chunk_size 131584 / hop 441 ≈ 299) + many chunks to mimic the main app's
            // 213-chunk run — catches autotune-on-real-shape and multi-run VRAM/workspace crashes.
            let frames = 299usize;
            eprintln!("[4] session created; running 40 inferences at frames={frames} ...");
            for i in 0..40 {
                let shape = vec![1i64, 2050, frames as i64, 2];
                let data = vec![0.0f32; 2050 * frames * 2];
                let value: ort::session::SessionInputValue =
                    ort::value::Tensor::from_array((shape, data.into_boxed_slice())).map_err(|e| format!("tensor: {e}"))?.into();
                let input_values: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> =
                    vec![(std::borrow::Cow::Borrowed("stft_repr"), value)];
                let _outputs = session.run(input_values).map_err(|e| format!("run {i}: {e}"))?;
                if i % 10 == 0 { eprintln!("  chunk {i} ok"); }
            }
            Ok(())
        })();
        match result {
            Ok(()) => eprintln!("[5] 40 RUNS OK — no crash with these options!"),
            Err(e) => eprintln!("[!] failed: {}", e),
        }
        let _ = tx.send(());
    });

    match rx.recv_timeout(Duration::from_secs(45)) {
        Ok(_) => eprintln!("[done]"),
        Err(_) => eprintln!("[TIMEOUT] DEADLOCKED at the last printed step"),
    }
    std::process::exit(0);
}
