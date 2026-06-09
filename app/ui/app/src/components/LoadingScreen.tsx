interface LoadingScreenProps {
  message?: string;
}

export default function LoadingScreen({
  message = "Loading...",
}: LoadingScreenProps) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#171717",
        color: "#ffffff",
        fontFamily:
          "ui-sans-serif, system-ui, -apple-system, sans-serif",
        fontSize: "16px",
        userSelect: "none",
        WebkitUserSelect: "none",
      }}
      onMouseDown={() => window.drag && window.drag()}
      onDoubleClick={() => window.doubleClick && window.doubleClick()}
    >
      <div>{message}</div>
    </div>
  );
}
