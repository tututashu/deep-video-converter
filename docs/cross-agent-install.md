# 在 Claude Code / Codex 中使用本 Skill

本 skill 遵循 **Agent Skills（SKILL.md）开放标准**：一个 `SKILL.md`（YAML frontmatter 声明 name + description）+ 可选 `references/` 资源目录。因此不止 WorkBuddy，**Claude Code、OpenAI Codex CLI** 等主流 agent 都能直接识别并使用，无需改写格式。

> 只需把本仓库内容放到对应工具的 skills 目录即可。目录结构即 skill：`SKILL.md` 必须在顶层。

## 一键安装

两个工具都用 git clone 到目标目录（仓库结构本身就是标准 skill 布局）：

### Claude Code

```bash
# 用户级（所有项目可用）
git clone https://github.com/tututashu/deep-video-converter-skill.git ~/.claude/skills/deep-video-converter

# 或项目级（仅该项目可用，随仓库提交）
git clone https://github.com/tututashu/deep-video-converter-skill.git <你的项目>/.claude/skills/deep-video-converter
```

### Codex CLI

```bash
# 用户级
git clone https://github.com/tututashu/deep-video-converter-skill.git ~/.codex/skills/deep-video-converter

# 或项目级（Codex 也读 .agents/skills/）
git clone https://github.com/tututashu/deep-video-converter-skill.git <你的项目>/.codex/skills/deep-video-converter
```

## 验证生效

新开会话后，直接自然语言描述任务，agent 会根据 `description` 自动匹配本 skill：

- "启动深度视频转换器"
- "帮我装并跑起来 deep-video-converter，把视频转成姿态骨架叠加"
- "跑一下它的 CLI 端到端验证"

**Claude Code**：描述匹配时自动加载；也可在提示中显式提及 skill 名 `deep-video-converter` 帮助命中。

**Codex CLI**：同样自动匹配；也可用 `$` 前缀点名调用（如 `$deep-video-converter`），或输入 `/skills` 浏览已发现的 skill 列表。若列表中看不到本 skill，重启 Codex 强制重新扫描。

## 卸载

```bash
rm -rf ~/.claude/skills/deep-video-converter   # Claude Code
rm -rf ~/.codex/skills/deep-video-converter    # Codex
```

## 注意事项

- **保持同步**：本 skill 随 [deep-video-converter-skill](https://github.com/tututashu/deep-video-converter-skill) 仓库更新。改动后在新会话中 `git -C ~/.claude/skills/deep-video-converter pull`（或重跑 clone）即可拿到最新版。
- **Codex 版本**：Codex 对 SKILL.md 的支持随 CLI 版本演进，如遇未识别请升级 Codex CLI；面向 Codex 专属增强（如 MCP 工具依赖）可另加可选 `openai.yaml`，其他 agent 会忽略该文件。
- **运行前提**：skill 本身只承载操作指引，真正运行 deep-video-converter 仍需本机满足前置条件（Python 3.11 + 3.12、ffmpeg，首次联网下载模型 ~450MB），详见主文件 `SKILL.md` 的「One-shot setup」。
- 本机同时使用 WorkBuddy 时，skill 源以 `~/.workbuddy/skills/deep-video-converter` 为准，各工具目录是它的副本。
