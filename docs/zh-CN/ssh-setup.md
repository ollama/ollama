# SSH 金鑰設置指南

本文檔說明如何為 Ollama 項目添加和配置 SSH 金鑰。

## 什麼是 SSH 金鑰？

SSH 金鑰是一種安全的身份驗證方式，用於在不使用密碼的情況下連接到遠程服務器或 Git 存儲庫。它包括一對金鑰：
- **公鑰** (Public Key)：可以安全地共享，添加到 GitHub 等服務
- **私鑰** (Private Key)：必須保密，僅保存在本地計算機上

## 生成新的 SSH 金鑰

### 1. 檢查現有的 SSH 金鑰

```bash
ls -al ~/.ssh
```

查找是否已經存在 `id_ed25519.pub` 或 `id_rsa.pub` 文件。

### 2. 生成新的 SSH 金鑰對

推薦使用 ED25519 算法，它更安全且性能更好：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

如果您的系統不支持 ED25519，可以使用 RSA：

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

按照提示操作：
- 按 Enter 接受默認文件位置
- 輸入安全的密碼短語（可選但推薦）
- 再次輸入密碼短語確認

### 3. 啟動 SSH 代理

```bash
eval "$(ssh-agent -s)"
```

### 4. 將 SSH 私鑰添加到代理

```bash
ssh-add ~/.ssh/id_ed25519
```

## 添加 SSH 公鑰到 GitHub

### 1. 複製公鑰到剪貼板

在 Linux 上：
```bash
cat ~/.ssh/id_ed25519.pub
```

然後手動複製輸出內容。

在 macOS 上：
```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

### 2. 添加到 GitHub

1. 登錄 GitHub
2. 點擊右上角的個人頭像，選擇 **Settings**
3. 在左側邊欄中，點擊 **SSH and GPG keys**
4. 點擊 **New SSH key** 或 **Add SSH key**
5. 在 "Title" 字段中，為新金鑰添加描述性標籤（例如："個人筆記本電腦"）
6. 將公鑰粘貼到 "Key" 字段中
7. 點擊 **Add SSH key**
8. 如果提示，請確認您的 GitHub 密碼

## 測試 SSH 連接

```bash
ssh -T git@github.com
```

您應該看到類似以下的消息：
```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

## 在項目中使用 SSH

### 克隆存儲庫

使用 SSH URL 克隆：
```bash
git clone git@github.com:ollama/ollama.git
```

### 更改現有存儲庫的遠程 URL

如果您已經使用 HTTPS 克隆了存儲庫，可以切換到 SSH：

```bash
git remote set-url origin git@github.com:ollama/ollama.git
```

驗證更改：
```bash
git remote -v
```

## GitHub Actions 中使用 SSH 金鑰

如果您需要在 GitHub Actions 工作流中使用 SSH 金鑰：

### 1. 生成專用的部署金鑰

```bash
ssh-keygen -t ed25519 -C "deploy-key@ollama" -f ~/.ssh/ollama_deploy_key -N ""
```

### 2. 添加部署金鑰到 GitHub

1. 進入存儲庫的 **Settings** → **Deploy keys**
2. 點擊 **Add deploy key**
3. 添加標題和公鑰內容
4. 如果需要寫入權限，勾選 **Allow write access**

### 3. 添加私鑰到 GitHub Secrets

1. 進入存儲庫的 **Settings** → **Secrets and variables** → **Actions**
2. 點擊 **New repository secret**
3. Name: `SSH_PRIVATE_KEY`
4. Value: 私鑰的完整內容（包括開始和結束標記）

### 4. 在工作流中使用

```yaml
- name: Setup SSH
  uses: webfactory/ssh-agent@v0.9.0
  with:
    ssh-private-key: ${{ secrets.SSH_PRIVATE_KEY }}
```

## 安全最佳實踐

1. **永遠不要共享您的私鑰** - 私鑰應該只存在於您的本地計算機上
2. **使用密碼短語保護私鑰** - 增加額外的安全層
3. **為不同的服務使用不同的金鑰** - 如果一個金鑰被洩露，其他服務仍然安全
4. **定期輪換金鑰** - 建議每年更新 SSH 金鑰
5. **刪除不再使用的金鑰** - 從 GitHub 和本地系統中移除舊的金鑰

## 故障排除

### 權限被拒絕

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

### SSH 代理問題

確保 SSH 代理正在運行：
```bash
eval "$(ssh-agent -s)"
ssh-add -l
```

### 連接超時

檢查您的 SSH 配置：
```bash
cat ~/.ssh/config
```

添加以下內容（如果不存在）：
```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
```

## 參考資料

- [GitHub SSH 文檔](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- [SSH 金鑰類型比較](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
