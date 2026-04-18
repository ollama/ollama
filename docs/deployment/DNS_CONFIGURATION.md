# 🔧 DNS Configuration for ollama.elevatediq.ai

## Quick Setup

Add this CNAME record to your DNS:

| Field | Value |
|-------|-------|
| **Name** | `ollama` |
| **Type** | `CNAME` |
| **Value** | `ghs.googlehosted.com` |
| **TTL** | `300` |

---

## Step-by-Step Instructions

### 1. Go to Your DNS Provider

- **AWS Route 53**: https://console.aws.amazon.com/route53/
- **Cloudflare**: https://dash.cloudflare.com/
- **GoDaddy**: https://dcc.godaddy.com/
- **Google Cloud DNS**: https://console.cloud.google.com/net-services/dns/zones
- **NameCheap**: https://www.namecheap.com/
- **BlueHost**: https://www.bluehost.com/
- Other provider: Log into your DNS management panel

### 2. Create CNAME Record

**AWS Route 53 Example**:
```
Record name: ollama.elevatediq.ai
Record type: CNAME
Value: ghs.googlehosted.com
TTL: 300
```

**Cloudflare Example**:
```
Type: CNAME
Name: ollama
Content: ghs.googlehosted.com
TTL: Auto (or 300)
Proxy status: DNS only
```

**GoDaddy Example**:
```
Type: CNAME
Name: ollama
Points to: ghs.googlehosted.com
TTL: 600
```

### 3. Save and Wait

DNS typically propagates within:
- ⚡ Fast: 5 minutes
- 📊 Normal: 15-30 minutes
- 🐢 Slow: Up to 2 hours (rare)

### 4. Verify

Check if DNS is propagated:

```bash
# Check DNS resolution
nslookup ollama.elevatediq.ai
# or
dig ollama.elevatediq.ai

# Should return: ghs.googlehosted.com
```

### 5. Test the Endpoint

Once DNS propagates:

```bash
# Test health check
curl https://ollama.elevatediq.ai/health

# Should return:
# {"status":"healthy","service":"ollama-api","version":"1.0.0"}
```

---

## Verify Propagation

Use these tools to check:

- **mxtoolbox**: https://mxtoolbox.com/mxlookup.aspx
- **whatsmydns**: https://www.whatsmydns.net/
- **dnschecker**: https://dnschecker.org/
- **digwebinterface**: https://www.digwebinterface.com/

Enter: `ollama.elevatediq.ai`  
Expected result: `ghs.googlehosted.com`

---

## Troubleshooting

### DNS Not Propagating

**Wait longer** (up to 2 hours, usually less)

**Check:**
1. Are you logged into the correct DNS account?
2. Did you save the record?
3. Is the subdomain correct (not the full domain)?
4. Is the target exactly `ghs.googlehosted.com`?

### Still Not Working

1. Clear browser cache: `Ctrl+Shift+Delete` (or `Cmd+Shift+Delete`)
2. Use different DNS: `1.1.1.1` or `8.8.8.8`
3. Try incognito/private window
4. Check with: `nslookup ollama.elevatediq.ai`

---

## Once DNS is Ready

Access the Ollama API at:

```
https://ollama.elevatediq.ai
```

**Available Endpoints**:
- `https://ollama.elevatediq.ai/health` - Health check
- `https://ollama.elevatediq.ai/api/v1/health` - API status
- `https://ollama.elevatediq.ai/docs` - Interactive documentation
- `https://ollama.elevatediq.ai/` - Root endpoint

---

## Fallback URLs

If DNS is taking too long, use these:

**Direct Cloud Run URL** (always works):
```
https://ollama-service-794896362693.us-central1.run.app
```

**Load Balancer Path** (if configured):
```
https://elevatediq.ai/ollama
```

---

## Support

For issues:
1. Check DNS propagation: https://whatsmydns.net/
2. Verify record in GCP: `gcloud run domain-mappings list --region=us-central1 --project=elevatediq`
3. Check Cloud Run logs: https://console.cloud.google.com/logs?project=elevatediq

---

**DNS Record Status**: ⏳ Pending Configuration  
**Service Status**: ✅ Live on Cloud Run  
**Production URL**: https://ollama.elevatediq.ai
