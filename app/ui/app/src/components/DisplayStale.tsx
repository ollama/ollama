import { Model } from "@/gotypes";
import { useSendMessage, useIsStreaming } from "@/hooks/useChats";
import { Display, type DisplayAction } from "@/components/ui/display";

interface DisplayStaleProps {
  model: Model;
  onDismiss: () => void;
  className?: string;
  chatId: string;
  onScrollToBottom?: () => void;
}

export const DisplayStale = ({
  model,
  onDismiss,
  className,
  chatId,
  onScrollToBottom,
}: DisplayStaleProps) => {
  const sendMessage = useSendMessage(chatId);
  const isStreaming = useIsStreaming(chatId);

  const handleUpdateModel = async () => {
    if (onScrollToBottom) {
      onScrollToBottom();
    }

    try {
      sendMessage.mutate({
        message: "",
        forceUpdate: true,
      });
    } catch (error) {
      console.error("Failed to update model:", error);
    }

    onDismiss();
  };

  const action: DisplayAction = {
    label: "Update",
    onClick: handleUpdateModel,
    disabled: isStreaming,
    gradientColors: "from-zinc-500/20 via-slate-500/20 to-gray-500/20",
  };

  return (
    <Display
      message={`A newer version of ${model.model} is available`}
      variant="zinc"
      onDismiss={onDismiss}
      action={action}
      className={className}
    />
  );
};
