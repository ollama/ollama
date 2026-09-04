import type { ErrorEvent } from "@/gotypes";
import { Display, type DisplayAction } from "@/components/ui/display";
import { useUser } from "@/hooks/useUser";

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
  const { connectUser, connectionError, isAuthenticated, isConnecting } =
    useUser();

  if (!error || error.code !== "cloud_unauthorized" || isAuthenticated)
    return null;

  const action: DisplayAction = {
    label: "Sign In",
    onClick: connectUser,
    disabled: isConnecting,
    loading: isConnecting,
  };

  return (
    <Display
      message={
        connectionError || message || "Cloud models require an Ollama account"
      }
      action={action}
      className={className}
      onDismiss={onDismiss}
    />
  );
};
