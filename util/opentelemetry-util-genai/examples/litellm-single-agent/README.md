though not actually instrumented with genai utils, here's an example app using litellm to showcase the native instrumentation

## Kubernetes Deployment

This example includes Kubernetes manifests to deploy litellm as a proxy service.

### Prerequisites

- A Kubernetes cluster (minikube, Docker Desktop, or cloud provider)
- `kubectl` CLI installed and configured
- An OpenAI API key

### Option 1: Using secret.yaml (Recommended for development)

1. Add your OpenAI API key to `secret.yaml`:
   ```bash
   # Encode your API key to base64
   echo -n "sk-proj-your-actual-key-here" | base64
   ```
   
   Copy the output and paste it into `secret.yaml`:
   ```yaml
   OPENAI_API_KEY: your-base64-encoded-key-here
   ```

2. Apply the secret and deployment:
   ```bash
   kubectl apply -f secret.yaml
   kubectl apply -f kub.yaml
   ```

3. Don't commit `secret.yaml` to git (it's in `.gitignore`)

### Option 2: Direct kubectl command (Recommended for security)

Create the secret directly without storing it in a file:

```bash
kubectl create secret generic litellm-secrets \
  --from-literal=OPENAI_API_KEY=sk-proj-your-actual-key-here
```

Then apply only the deployment:
```bash
kubectl apply -f kub.yaml
```

### Verify the deployment

```bash
# Check pod status
kubectl get pods -l app=litellm

# View logs
kubectl logs -l app=litellm

# Port forward to access the service
kubectl port-forward svc/litellm 4000:4000
```

The litellm proxy will be available at `http://localhost:4000`

