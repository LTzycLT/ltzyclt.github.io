# 项目经验总结

## 2025-03-05: 修复 GitHub Pages tag/category 页面 404

### 问题
- GitHub Pages 报错 "Page Not Found"，特别是 `/tags/acm/` 页面
- 原因：GitHub Pages 不支持 `jekyll-archives` 插件，导致 tag/category 页面无法生成

### 解决方案
1. 创建 GitHub Actions 工作流 `.github/workflows/jekyll.yml`，使用 bundler 构建 Jekyll 网站
2. 工作流会自动安装插件并生成 tag/category 页面
3. 构建产物推送到 gh-pages 分支
4. GitHub Pages 设置 Source 为 "gh-pages branch"（或 "GitHub Actions"）

### 配置要点
- `_config.yml` 中配置 `jekyll-archives` 插件（tags 和 categories）
- 需要对应的 layout 文件（如 `archive-taxonomy.html`）
- GitHub Actions 工作流使用 `peaceiris/actions-gh-pages@v4` 部署

### 注意事项
- 不要依赖 GitHub Pages 默认构建，因为它不支持所有 Jekyll 插件
- 使用自定义工作流可以绕过这个限制
- 每次 push 到 master 分支会自动触发构建和部署