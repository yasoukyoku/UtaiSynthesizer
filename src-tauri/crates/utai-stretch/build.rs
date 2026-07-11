fn main() {
    // The hot loop lives in the C++ library — force -O2/-O3 regardless of the cargo profile
    // (cc otherwise mirrors the dev profile's opt-level 0, which would make dev-build stretching
    // 10-30x slower; same rationale as the utai-dsp opt-3 override).
    cc::Build::new()
        .cpp(true)
        .file("src/wrapper.cpp")
        .include("vendor")
        .flag_if_supported("/std:c++17")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("/EHsc")
        .opt_level(3)
        .debug(false)
        .compile("utai_stretch_cpp");
    println!("cargo:rerun-if-changed=src/wrapper.cpp");
    println!("cargo:rerun-if-changed=vendor/signalsmith-stretch/signalsmith-stretch.h");
}
