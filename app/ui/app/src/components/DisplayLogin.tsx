import type { ErrorEvent } from "@/gotypes";
import { Display, type DisplayAction } from "@/components/ui/display";
import { useUser } from "@/hooks/useUser";
import { useEffect, useState } from "react";

interface DisplayLoginProps {
  error: ErrorEvent | null;
  className?: string;
  onDismiss?: () => void;
  message?: string;
}

export const DisplayLogin = ({
  error,
  className,
  onDismiss,
  message,
}: DisplayLoginProps) => {
  const { fetchConnectUrl, refetchUser, isAuthenticated } = useUser();
  const [isAwaitingAuth, setIsAwaitingAuth] = useState(false);

  useEffect(() => {
    const handleFocus = () => {
      if (isAwaitingAuth) {
        setIsAwaitingAuth(false);
        refetchUser();
      }
    };

    window.addEventListener("focus", handleFocus);

    return () => {
      window.removeEventListener("focus", handleFocus);
    };
  }, [isAwaitingAuth, refetchUser]);

  useEffect(() => {
    if (isAuthenticated && isAwaitingAuth) {
      setIsAwaitingAuth(false);
      if (onDismiss) {
        onDismiss();
      }
    }
  }, [isAuthenticated, isAwaitingAuth, onDismiss]);

  if (!error || error.code !== "cloud_unauthorized" || isAuthenticated)
    return null;

  const handleSignIn = async () => {
    try {
      const { data: connectUrl } = await fetchConnectUrl();
      if (connectUrl) {
        window.open(connectUrl, "_blank");
        setIsAwaitingAuth(true);
      }
    } catch (error) {
      console.error("Error getting connect URL:", error);
    }
  };

  const action: DisplayAction = {
    label: "Sign In",
    onClick: handleSignIn,
  };

  return (
    <Display
      message={message || "Cloud models require an Ollama account"}
      action={action}
      className={className}
      onDismiss={onDismiss}
    />
  );
};
