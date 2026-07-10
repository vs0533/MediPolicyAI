# 腾讯云 Docker Compose 部署

本部署方案不使用镜像仓库。GitHub Actions 在推送 `v*.*.*` 版本 tag 或手动触发时，通过 SSH 上传源码包到腾讯云服务器，并在服务器本地执行 `docker compose build` 和 `docker compose up -d`。

## GitHub Secrets

必须配置：

- `TENCENT_HOST`：腾讯云公网 IP 或域名
- `SSH_PRIVATE_KEY`：可登录服务器的私钥

可选配置：

- `TENCENT_USER`：SSH 用户，默认 `root`
- `TENCENT_PORT`：SSH 端口，默认 `22`
- `TENCENT_DEPLOY_PATH`：部署目录，默认 `~/medipolicyai`

如果部署日志出现 `Permission denied (publickey,password)`，通常是 `SSH_PRIVATE_KEY` 和服务器用户不匹配。请确认私钥对应目标用户的 `~/.ssh/authorized_keys`，并配置正确的 `TENCENT_USER`，例如 `root`、`ubuntu` 或你的实际登录用户。

如果希望部署到 `/opt/medipolicyai`，需要先在服务器上创建目录并授权给部署用户：

```bash
sudo mkdir -p /opt/medipolicyai
sudo chown -R 当前部署用户:当前部署用户 /opt/medipolicyai
```

## 服务器目录

部署后目录结构：

```text
~/medipolicyai/
  current/              # 当前代码，由 Actions 自动更新
  current.prev/         # 上一个版本，便于手动回滚
  shared/
    .env                # 生产环境配置，不进入 Git
    data/               # Sirchmunk 工作目录、缓存、历史、提问统计
    policy-docs/        # 医保政策文档，只读挂载到容器
```

## 首次部署

首次运行 workflow 时，如果服务器没有 `/opt/medipolicyai/shared/.env`，会自动生成模板并主动失败。登录服务器补齐 `LLM_API_KEY` 后，重新运行 workflow。

```bash
ssh root@你的服务器
vim ~/medipolicyai/shared/.env
```

至少需要配置：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=你的密钥
LLM_MODEL_NAME=gpt-5.2
```

把政策文件放到：

```bash
~/medipolicyai/shared/policy-docs
```

然后重新运行 GitHub Actions 的“部署到腾讯云”workflow。

## 发布新版本

生产部署由版本 tag 触发，推荐使用语义化版本号：

```bash
git tag v1.0.0
git push origin v1.0.0
```

如果只是想验证部署链路，也可以在 GitHub Actions 页面手动运行“部署到腾讯云”workflow。

## 常用运维命令

```bash
cd ~/medipolicyai/current
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f
docker compose -f docker-compose.prod.yml restart
```

## 回滚

如果新版本异常，可以在服务器上执行：

```bash
cd ~/medipolicyai
rm -rf current.bad
mv current current.bad
mv current.prev current
cd current
docker compose -f docker-compose.prod.yml up -d --build
```

## 访问

容器默认监听宿主机 `8584` 端口：

```text
http://服务器IP:8584/
```

生产环境建议在腾讯云安全组只开放必要端口，并使用 Nginx 或腾讯云负载均衡配置 HTTPS 反向代理到 `127.0.0.1:8584`。
