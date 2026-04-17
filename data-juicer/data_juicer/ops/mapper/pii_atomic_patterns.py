from __future__ import annotations

# Shared regex fragments for atomic PII-cleaning mappers.

PATH_UNIX_PATTERN = (
    r"(^|[\s\"'(\[=])"
    r"(/(?:Users?|home|tmp|etc|var|opt|Applications)[^\s\"')\]]*(?:/[^\s\"')\]]*)*)"
)
PATH_WIN_PATTERN = r"(^|[\s\"'(\[=])([A-Za-z]:\\[^\s\"')\]]*(?:\\[^\s\"')\]]*)*)"
PATH_WIN_UNC_PATTERN = r"(^|[\s\"'(\[=])(\\\\[^\s\"')\]]+)"

PHONE_CN_PATTERN = r"\b1[3-9]\d{9}\b"
PHONE_INTL_PATTERN = r"\+\d{1,4}[-.\s]?\d{6,14}\b"
ID_CARD_CN_PATTERN = (
    r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"
)

SECRET_KV_PATTERN = (
    r"(\b(?:api[_-]?key|apikey|secret|password|passwd|token|auth|authorization"
    r"|credential|license[_-]?key)\s*[:=]\s*[\"']?)([^\s\"',}\]]+)"
)

CHANNEL_KV_PATTERN = (
    r"(\bchannel\s*[:=]\s*[\"']?|当前的\s*channel\s*[:：]\s*)"
    r"(feishu|dingtalk|wecom|wechat_work|email|mail|飞书|钉钉|企业微信|邮箱)\b"
)
FEISHU_OPEN_ID_PATTERN = r"\bou_[0-9a-f]{32}\b"
PLATFORM_OPEN_ID_PATTERN = r"\b(ou_|u_|uid_)[0-9a-zA-Z_-]{16,64}\b"

JWT_PATTERN = r"\beyJ[A-Za-z0-9_=-]+\.[A-Za-z0-9_=-]+\.[A-Za-z0-9_=-]+\b"
PEM_PATTERN = r"-----BEGIN [A-Z0-9 -]+-----[\s\S]*?-----END [A-Z0-9 -]+-----"
MAC_PATTERN = r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
