# 服务器端改动说明：为 evaluate.py 增加 Anthropic 原生 API（Lanz 网关）支持

> 把这份说明复制到服务器上的 Claude Code，让它执行下面 4 个文件的修改。
> 所有路径都是服务器上的绝对路径（Linux），根目录是 `/data1/ESR-GRPO/ESR-GRPO/exper/arex_bcplus_native/`。

## 背景

我们要把 AREX-Turbo 在 BrowseComp-Plus 上跑出来的推理结果交给一个 Judge 模型打分。Judge 走公司内部的 **Lanz 网关**，它是**原生 Anthropic 协议**（`/v1/messages` + `x-api-key` header），不是 OpenAI 格式。当前 evaluate.py 只支持 OpenAI 格式，需要加一个 Anthropic 分支。

Lanz 网关信息：
- 真正的 Messages API 上游地址：`http://lanz.hikvision.com/v3/anthropic/model/v1/messages`
- 所以 `AREX_JUDGE_BASE_URL` 应设为 `http://lanz.hikvision.com/v3/anthropic/model`（代码会自动拼上 `/v1/messages`）
- 认证用 `x-api-key` header（不是 `Authorization: Bearer`）
- 模型名：`Lanz-Auto`

**重要**：evaluate.py 的 Judge 请求是 Python urllib 直连 Lanz，**不经过** `127.0.0.1:18080` 那个兼容代理。因为 evaluate.py 自己构造请求，不会带 `?beta=true` 和 `Authorization: Bearer`，直接发 `x-api-key` + `/v1/messages` 正好是 Lanz 要的格式，所以 Judge 链路直连即可。

---

## 改动 1：`clients.py` — 新增 `_request_json_with_headers` 函数

**文件**：`/data1/ESR-GRPO/ESR-GRPO/exper/arex_bcplus_native/clients.py`

**原因**：现有 `_request_json` 把 `Authorization: Bearer` 写死在函数里，无法支持 Anthropic 的 `x-api-key` header。需要新增一个接受自定义 headers 的版本。

**操作**：找到现有的 `_request_json` 函数（文件开头第 14 行附近），把它改成调用新函数 `_request_json_with_headers`，并在其后新增 `_request_json_with_headers` 函数。

**把这一段**（现有的 `_request_json` 函数）：

```python
def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "arex-bcplus-native/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc
```

**替换成**：

```python
def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "arex-bcplus-native/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return _request_json_with_headers(url, payload=payload, headers=headers, timeout=timeout)


def _request_json_with_headers(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str],
    timeout: float = 120.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc
```

**要点**：
- `_request_json` 保持向后兼容（OpenAI 路径继续用它，自动加 `Authorization: Bearer`）。
- 新增的 `_request_json_with_headers` 接受完全自定义的 headers，Anthropic 路径用它来传 `x-api-key`。
- 两个函数共用同一段 urllib 请求逻辑，只是 header 来源不同。

---

## 改动 2：`evaluate.py` — 新增 `--judge-provider` 参数 + Anthropic 分支

**文件**：`/data1/ESR-GRPO/ESR-GRPO/exper/arex_bcplus_native/evaluate.py`

**原因**：让 evaluate.py 能根据 `--judge-provider` 选择走 OpenAI 还是 Anthropic 协议。

### 改动 2a：修改 import 行

**把**（文件第 14 行）：

```python
from clients import _request_json
```

**替换成**：

```python
from clients import _request_json, _request_json_with_headers
```

### 改动 2b：新增 `--judge-provider` 参数

**把**（`main()` 里 argparse 部分，第 109-111 行附近）：

```python
    parser.add_argument("--judge-base-url", default=os.getenv("AREX_JUDGE_BASE_URL"))
    parser.add_argument("--judge-model", default=os.getenv("AREX_JUDGE_MODEL", "Qwen3-32B"))
    parser.add_argument("--judge-api-key", default=os.getenv("AREX_JUDGE_API_KEY", "EMPTY"))
```

**替换成**：

```python
    parser.add_argument("--judge-base-url", default=os.getenv("AREX_JUDGE_BASE_URL"))
    parser.add_argument("--judge-model", default=os.getenv("AREX_JUDGE_MODEL", "Qwen3-32B"))
    parser.add_argument("--judge-api-key", default=os.getenv("AREX_JUDGE_API_KEY", "EMPTY"))
    parser.add_argument(
        "--judge-provider",
        default=os.getenv("AREX_JUDGE_PROVIDER", "openai"),
        choices=["openai", "anthropic"],
        help="Judge API 格式：openai（/chat/completions）或 anthropic（/v1/messages）",
    )
```

### 改动 2c：Judge 调用处分支

**把**（Judge 调用那段，第 157-159 行附近，`else:` 分支里构造 prompt 之后直接调 `_request_json` 的部分）：

```python
            prompt = GRADER_TEMPLATE.format(
                question=question, response=response, correct_answer=answer
            )
            raw = _request_json(
                f"{args.judge_base_url.rstrip('/')}/chat/completions",
                payload={
                    "model": args.judge_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": args.judge_max_tokens,
                },
                api_key=args.judge_api_key,
                timeout=300.0,
            )
            text = str(raw["choices"][0]["message"].get("content") or "")
            judged = parse_judgement(text)
            judged["raw_response"] = raw
            atomic_write_json(cache_path, judged)
```

**替换成**：

```python
            prompt = GRADER_TEMPLATE.format(
                question=question, response=response, correct_answer=answer
            )
            if args.judge_provider == "anthropic":
                # Anthropic 原生 API：/v1/messages，x-api-key header，响应在 content[0].text
                raw = _request_json_with_headers(
                    f"{args.judge_base_url.rstrip('/')}/v1/messages",
                    payload={
                        "model": args.judge_model,
                        "max_tokens": args.judge_max_tokens,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": args.judge_api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    timeout=300.0,
                )
                content_blocks = raw.get("content", [])
                text = str(content_blocks[0].get("text", "")) if content_blocks else ""
            else:
                # OpenAI 兼容：/chat/completions，Authorization Bearer，响应在 choices[0].message.content
                raw = _request_json(
                    f"{args.judge_base_url.rstrip('/')}/chat/completions",
                    payload={
                        "model": args.judge_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": args.judge_max_tokens,
                    },
                    api_key=args.judge_api_key,
                    timeout=300.0,
                )
                text = str(raw["choices"][0]["message"].get("content") or "")
            judged = parse_judgement(text)
            judged["raw_response"] = raw
            atomic_write_json(cache_path, judged)
```

**要点**：
- Anthropic 分支用 `/v1/messages`、`x-api-key` + `anthropic-version` header、从 `raw["content"][0]["text"]` 取文本。
- OpenAI 分支保持原样。
- 两分支后共用 `parse_judgement(text)` 解析 Judge 返回的结构化文本。

---

## 改动 3：`setup_env.sh` — 新增 `AREX_JUDGE_PROVIDER` 变量

**文件**：`/data1/ESR-GRPO/ESR-GRPO/exper/arex_bcplus_native/setup_env.sh`

**原因**：集中配置 Judge，支持 OpenAI / Anthropic 两种格式切换。

### 改动 3a：替换 Judge 配置段

**把**（Judge 配置段，第 54 行附近 `# Judge：不下载 Qwen3-32B...` 开始到 `fi` 结束）：

```bash
# Judge：不下载 Qwen3-32B（~64GB），改用外部 API。
# 未配置 base URL 或 API key 时，自动回退到精确匹配，只跑通评测链路，结果不作正式 Accuracy。
AREX_JUDGE_BASE_URL="${AREX_JUDGE_BASE_URL:-}"
AREX_JUDGE_MODEL="${AREX_JUDGE_MODEL:-glm-4}"
AREX_JUDGE_API_KEY="${AREX_JUDGE_API_KEY:-}"
if [ -z "$AREX_JUDGE_BASE_URL" ] || [ -z "$AREX_JUDGE_API_KEY" ]; then
  AREX_EXACT_MATCH_SMOKE=1
  _JUDGE_MODE="精确匹配（未配置 Judge，结果不作正式 Accuracy）"
else
  AREX_EXACT_MATCH_SMOKE="${AREX_EXACT_MATCH_SMOKE:-0}"
  _JUDGE_MODE="Judge: $AREX_JUDGE_MODEL"
fi
```

> 注意：上面这段是**服务器上当前的旧版本**。如果服务器上的版本和这里不完全一样（比如注释行数不同），以 `AREX_JUDGE_BASE_URL=` 这一行到 `fi` 这一段为准替换。

**替换成**：

```bash
# Judge：不下载 Qwen3-32B（~64GB），改用外部 API。支持两种格式：
#
# 1) OpenAI 兼容（GLM、智谱云、one-api 网关等）：
#    export AREX_JUDGE_PROVIDER=openai
#    export AREX_JUDGE_BASE_URL=https://open.bigmodel.cn/api/paas/v4
#    export AREX_JUDGE_API_KEY=你的key
#    export AREX_JUDGE_MODEL=glm-4
#
# 2) Anthropic 原生 API（Claude 网关）：
#    export AREX_JUDGE_PROVIDER=anthropic
#    export AREX_JUDGE_BASE_URL=https://api.anthropic.com
#    export AREX_JUDGE_API_KEY=你的anthropic key
#    export AREX_JUDGE_MODEL=claude-3-5-sonnet-20241022
#
# 未配置 base URL 或 API key 时，自动回退到精确匹配，只跑通评测链路，结果不作正式 Accuracy。
AREX_JUDGE_PROVIDER="${AREX_JUDGE_PROVIDER:-openai}"
AREX_JUDGE_BASE_URL="${AREX_JUDGE_BASE_URL:-}"
AREX_JUDGE_MODEL="${AREX_JUDGE_MODEL:-glm-4}"
AREX_JUDGE_API_KEY="${AREX_JUDGE_API_KEY:-}"
if [ -z "$AREX_JUDGE_BASE_URL" ] || [ -z "$AREX_JUDGE_API_KEY" ]; then
  AREX_EXACT_MATCH_SMOKE=1
  _JUDGE_MODE="精确匹配（未配置 Judge，结果不作正式 Accuracy）"
else
  AREX_EXACT_MATCH_SMOKE="${AREX_EXACT_MATCH_SMOKE:-0}"
  _JUDGE_MODE="$AREX_JUDGE_PROVIDER Judge: $AREX_JUDGE_MODEL"
fi
```

**要点**：
- 新增 `AREX_JUDGE_PROVIDER` 变量，默认 `openai`。
- `_JUDGE_MODE` 显示里加上 provider 前缀，方便 source 时确认走的是哪条链路。

### 改动 3b：export 行补上 `AREX_JUDGE_PROVIDER`

**把**（export 段第 97 行附近）：

```bash
export AREX_JUDGE_BASE_URL AREX_JUDGE_MODEL AREX_JUDGE_API_KEY AREX_EXACT_MATCH_SMOKE
```

**替换成**：

```bash
export AREX_JUDGE_PROVIDER AREX_JUDGE_BASE_URL AREX_JUDGE_MODEL AREX_JUDGE_API_KEY AREX_EXACT_MATCH_SMOKE
```

---

## 改动 4：`run_experiment.sh` — EVAL_ARGS 透传 `--judge-provider`

**文件**：`/data1/ESR-GRPO/ESR-GRPO/exper/arex_bcplus_native/run_experiment.sh`

**原因**：这是**关键接线**。run_experiment.sh 组装 `EVAL_ARGS` 时如果不传 `--judge-provider`，evaluate.py 会用 argparse 默认值 `openai`，即使 setup_env.sh 里设了 `AREX_JUDGE_PROVIDER=anthropic` 也不会生效，Judge 永远走 OpenAI 格式打到 Lanz 的 `/chat/completions` 然后失败。

**把**（第 77-87 行附近，组装 `EVAL_ARGS` 的 else 分支）：

```bash
  : "${AREX_JUDGE_BASE_URL:?set AREX_JUDGE_BASE_URL or set AREX_EXACT_MATCH_SMOKE=1}"
  EVAL_ARGS=(
    --judge-base-url "${AREX_JUDGE_BASE_URL}"
    --judge-model "${AREX_JUDGE_MODEL:-Qwen3-32B}"
    --judge-api-key "${AREX_JUDGE_API_KEY:-EMPTY}"
  )
```

**替换成**：

```bash
  : "${AREX_JUDGE_BASE_URL:?set AREX_JUDGE_BASE_URL or set AREX_EXACT_MATCH_SMOKE=1}"
  EVAL_ARGS=(
    --judge-provider "${AREX_JUDGE_PROVIDER:-openai}"
    --judge-base-url "${AREX_JUDGE_BASE_URL}"
    --judge-model "${AREX_JUDGE_MODEL:-Qwen3-32B}"
    --judge-api-key "${AREX_JUDGE_API_KEY:-EMPTY}"
  )
```

**要点**：只在 `EVAL_ARGS` 数组里加一行 `--judge-provider`，其余不动。`${AREX_JUDGE_PROVIDER:-openai}` 保证未设置时回退到 openai，向后兼容。

---

## 改完后验证（在服务器上执行）

四个文件都改完后，跑这三条命令确认无误：

```bash
cd /data1/ESR-GRPO/ESR-GRPO/exper/arex_bcplus_native

# 1. Python 编译检查
python -m compileall -q clients.py evaluate.py && echo "compile OK"

# 2. shell 语法检查
bash -n setup_env.sh && bash -n run_experiment.sh && echo "bash OK"

# 3. 单元测试（应看到 Ran 11 tests OK）
python -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -5

# 4. 确认 --judge-provider 参数已注册
python evaluate.py --help 2>&1 | grep -A1 "judge-provider"
```

**预期输出**：
- `compile OK`
- `bash OK`
- `Ran 11 tests in ... OK`
- `--judge-provider {openai,anthropic}` 出现在 help 里

四项全过即改动完成。

---

## 改完后的配置与运行（供后续使用，不属于本次改动）

改动验证通过后，配置 Lanz Judge 并跑实验：

```bash
cd /data1/ESR-GRPO/ESR-GRPO/exper/arex_bcplus_native

# 配置 Lanz 作为 Judge（直连，不走 18080 代理）
export AREX_JUDGE_PROVIDER=anthropic
export AREX_JUDGE_BASE_URL=http://lanz.hikvision.com/v3/anthropic/model
export AREX_JUDGE_API_KEY=<你的Lanz key>
export AREX_JUDGE_MODEL=Lanz-Auto

source setup_env.sh
# 应显示：Judge 模式: anthropic Judge: Lanz-Auto

# 跑 1 题 smoke 验证全链路
AREX_LIMIT=1 bash run_experiment.sh
```
