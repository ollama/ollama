# SSH Key Setup Guide

This document explains how to add and configure SSH keys for the Ollama project.

## What is an SSH Key?

SSH keys are a secure authentication method used to connect to remote servers or Git repositories without using passwords. They consist of a key pair:
- **Public Key**: Can be safely shared and added to services like GitHub
- **Private Key**: Must be kept secret and only stored on your local computer

## Generating a New SSH Key

### 1. Check for Existing SSH Keys

```bash
ls -al ~/.ssh
```

Look for existing `id_ed25519.pub` or `id_rsa.pub` files.

### 2. Generate a New SSH Key Pair

We recommend using the ED25519 algorithm as it's more secure and performant:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

If your system doesn't support ED25519, use RSA:

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

Follow the prompts:
- Press Enter to accept the default file location
- Enter a secure passphrase (optional but recommended)
- Re-enter the passphrase to confirm

### 3. Start the SSH Agent

```bash
eval "$(ssh-agent -s)"
```

### 4. Add Your SSH Private Key to the Agent

```bash
ssh-add ~/.ssh/id_ed25519
```

## Adding Your SSH Public Key to GitHub

### 1. Copy the Public Key to Clipboard

On Linux:
```bash
cat ~/.ssh/id_ed25519.pub
```

Then manually copy the output.

On macOS:
```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

### 2. Add to GitHub

1. Log in to GitHub
2. Click your profile photo in the upper-right corner, then select **Settings**
3. In the sidebar, click **SSH and GPG keys**
4. Click **New SSH key** or **Add SSH key**
5. In the "Title" field, add a descriptive label (e.g., "Personal Laptop")
6. Paste your public key into the "Key" field
7. Click **Add SSH key**
8. If prompted, confirm your GitHub password

## Test Your SSH Connection

```bash
ssh -T git@github.com
```

You should see a message similar to:
```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

## Using SSH in Your Project

### Clone a Repository

Clone using the SSH URL:
```bash
git clone git@github.com:ollama/ollama.git
```

### Change Existing Repository Remote URL

If you've already cloned a repository using HTTPS, you can switch to SSH:

```bash
git remote set-url origin git@github.com:ollama/ollama.git
```

Verify the change:
```bash
git remote -v
```

## Using SSH Keys in GitHub Actions

If you need to use SSH keys in GitHub Actions workflows:

### 1. Generate a Dedicated Deploy Key

```bash
ssh-keygen -t ed25519 -C "deploy-key@ollama" -f ~/.ssh/ollama_deploy_key -N ""
```

### 2. Add Deploy Key to GitHub

1. Go to repository **Settings** → **Deploy keys**
2. Click **Add deploy key**
3. Add a title and paste the public key content
4. Check **Allow write access** if needed

### 3. Add Private Key to GitHub Secrets

1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `SSH_PRIVATE_KEY`
4. Value: The complete private key content (including begin/end markers)

### 4. Use in Workflow

```yaml
- name: Setup SSH
  uses: webfactory/ssh-agent@v0.9.0
  with:
    ssh-private-key: ${{ secrets.SSH_PRIVATE_KEY }}
```

## Security Best Practices

1. **Never share your private key** - The private key should only exist on your local computer
2. **Protect private keys with passphrases** - Adds an extra layer of security
3. **Use different keys for different services** - If one key is compromised, other services remain secure
4. **Regularly rotate keys** - Recommended to update SSH keys annually
5. **Remove unused keys** - Delete old keys from both GitHub and your local system

## Troubleshooting

### Permission Denied

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

### SSH Agent Issues

Ensure the SSH agent is running:
```bash
eval "$(ssh-agent -s)"
ssh-add -l
```

### Connection Timeout

Check your SSH configuration:
```bash
cat ~/.ssh/config
```

Add the following (if not present):
```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
```

## References

- [GitHub SSH Documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- [SSH Key Type Comparison](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
