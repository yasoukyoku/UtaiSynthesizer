# Security Policy / 安全政策

## Reporting a vulnerability / 报告安全漏洞

If you find a security vulnerability in UtaiSynthesizer (the app, its updater, or its
model/asset download pipeline), please **do not open a public issue**. Instead use GitHub's
private vulnerability reporting:

**[Report a vulnerability](https://github.com/yasoukyoku/UtaiSynthesizer/security/advisories/new)**

如果你发现了 UtaiSynthesizer(应用本体、更新器、或模型/资产下载链)的安全漏洞,请**不要**发公开
issue,改用上面的 GitHub 私密漏洞报告入口。

Please include: affected version, reproduction steps, and impact assessment.
请附上:受影响版本、复现步骤、影响评估。

## Scope notes / 范围说明

- The app runs fully locally; it makes network requests only for: update checks (GitHub
  Releases), model/asset downloads (Hugging Face / GitHub and their mirrors), and the
  optional mirror connectivity test in Settings. Downloads are sha256-verified; updates are
  minisign-verified.
- 应用完全本地运行;仅在检查更新(GitHub Releases)、下载模型/资产(Hugging Face / GitHub 及镜像)、
  以及设置中的镜像连通性测试时联网。下载有 sha256 校验;更新有 minisign 签名校验。
