# Claude Code Instructions

- **Git commits**: Use the default git user, not Claude's Co-Authored-By signature
- **Project structure**: Rust monorepo with WASM modules
  - Libraries in `libs/`
  - WASM apps build with `wasm-pack build --target web --out-dir www/pkg`
