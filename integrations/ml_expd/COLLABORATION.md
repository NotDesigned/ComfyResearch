# ComfyResearch 与 ML-Expd 协作约定

两边按仓库分工。本文件记录用户确定的规则；改动规则需要用户同意。

## 分工

| 负责方 | 负责的事 | 不做的事 |
| --- | --- | --- |
| Mac agent | ComfyResearch 的开发、测试和本地浏览器验收，**合并 ComfyResearch 的 PR** | 不改 ML-Expd |
| 服务端 agent | ML-Expd 服务端、`ml-exp` 客户端、发版部署和生产环境；审查和实测 ComfyResearch 里调用 `ml-exp` 的代码 | 不合并 ComfyResearch 的 PR |

服务器上的 Codex 不再参与 ComfyResearch。服务端的 `/root/comfyresearch` 克隆只用于审查和验证。

## 范围

ComfyResearch 这边只做一件事：把图和代码通过 `ml-exp` 提交出去，并读回结果。发现 `ml-exp` 本身可以设计得更好时，作为需求提给 ML-Expd，不在 ComfyResearch 里绕过去。

## 沟通

- **直接对话：** server-tokyo 上的 `/root/comfyresearch-mac.md`（Mac agent 写）和 `/root/comfyresearch-mac.reply.md`（服务端 agent 写）。`/root/client-feedback` 那一对文件属于 slowrun，不要写入。
- **需要长期跟踪的事项用 GitHub issue：**
  - 给 Mac agent 的需求：[NotDesigned/ComfyResearch](https://github.com/NotDesigned/ComfyResearch/issues)，标签 `ml-exp`。
  - 给 ML-Expd 的需求：[NotDesigned/ml-experiment-control](https://github.com/NotDesigned/ml-experiment-control/issues)，标签 `comfyresearch`。
  - 涉及两边改动时，在 issue 和 PR 里互相链接。
- 服务端发版改变 CLI JSON 字段、结果包目录或文件名等接口时，写进 release notes，并在 ComfyResearch 开 issue 提醒。

## Git

1. 一件事一个分支、一个 PR，目标是 fork 的 `main`。CI 通过后再合并。
2. 不 force push，不改写对方的分支。需要在对方分支上补东西时，只做快进推送，并在 PR 里留言说明。
3. 不向上游 MetaCircleAI 推送或提 PR。是否提上游由用户决定。
4. 每次开工前先 `git fetch`。

## 测试

- 测试以真实的 `ml-exp` 输出和真实的结果包为准，不手写假的数据结构。"Load results 一定失败"就是因为测试自己构造了结果包，才没有被发现。
- 需要客户端行为的测试使用 CI 固定的 `ml-exp` 二进制（`COMFYRESEARCH_TEST_ML_EXP_BINARY`，版本和 SHA-256 写在 CI 里）。客户端发新版时，由服务端 agent 提 PR 更新版本和校验值。
- 改动提交或读取结果的 PR，合并前在 `pai-native-cpu` 上跑一次真实的"提交 → 读取结果"。Mac agent 从工作台提交，或请服务端 agent 运行端到端验证。

不单独建开发用的 ML-Expd 项目和 token。
