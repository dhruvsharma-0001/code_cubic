export const CLOUD_AUTH_SESSION_KEY = "edgemind_cloud_auth_session";
export const CLOUD_AUTH_EVENT_NAME = "edgemind_cloud_auth_change";

export function isCloudConsoleAuthenticated(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.sessionStorage.getItem(CLOUD_AUTH_SESSION_KEY) === "authorized";
  } catch {
    return false;
  }
}

export async function authenticateCloudConsole(candidatePassword: string): Promise<boolean> {
  if (typeof window === "undefined") return false;
  
  // Verify with backend
  try {
    const res = await fetch("http://localhost:8000/health", {
      headers: {
        "X-EdgeMind-API-Key": candidatePassword
      }
    });
    if (res.ok) {
      window.sessionStorage.setItem(CLOUD_AUTH_SESSION_KEY, "authorized");
      // Store key to use for API requests
      window.sessionStorage.setItem("edgemind_api_key", candidatePassword);
      window.dispatchEvent(new Event(CLOUD_AUTH_EVENT_NAME));
      return true;
    }
  } catch {
    return false;
  }
  return false;
}

export function lockCloudConsole(): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(CLOUD_AUTH_SESSION_KEY);
    window.sessionStorage.removeItem("edgemind_api_key");
    window.dispatchEvent(new Event(CLOUD_AUTH_EVENT_NAME));
  } catch {}
}
