import * as React from "react";

export interface SliderProps {
  label?: string;
  options?: { value: number; label: string }[];
  value?: number;
  onChange?: (value: number) => void;
  className?: string;
}

const Slider = React.forwardRef<HTMLDivElement, SliderProps>(
  ({ label, options, value = 0, onChange }, ref) => {
    const [selectedValue, setSelectedValue] = React.useState(value);
    const [isDragging, setIsDragging] = React.useState(false);
    const containerRef = React.useRef<HTMLDivElement>(null);

    // Update internal state when value prop changes
    React.useEffect(() => {
      setSelectedValue(value);
    }, [value]);

    const handleClick = (optionValue: number) => {
      setSelectedValue(optionValue);
      onChange?.(optionValue);
    };

    const getClosestOption = (clientX: number) => {
      if (!containerRef.current || !options) return null;

      const rect = containerRef.current.getBoundingClientRect();
      const relativeX = clientX - rect.left;
      const width = rect.width;
      const segmentWidth = width / (options.length - 1);

      let closestIndex = Math.round(relativeX / segmentWidth);
      closestIndex = Math.max(0, Math.min(closestIndex, options.length - 1));

      return options[closestIndex].value;
    };

    const handleMouseDown = (e: React.MouseEvent) => {
      setIsDragging(true);
      e.preventDefault();
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;

      const closestValue = getClosestOption(e.clientX);
      if (closestValue !== null && closestValue !== selectedValue) {
        setSelectedValue(closestValue);
        // Don't call onChange during drag, just update visual state
      }
    };

    const handleMouseUp = () => {
      if (isDragging) {
        // Call onChange with the final value when drag ends
        onChange?.(selectedValue);
      }
      setIsDragging(false);
    };

    React.useEffect(() => {
      if (isDragging) {
        document.addEventListener("mousemove", handleMouseMove);
        document.addEventListener("mouseup", handleMouseUp);
        return () => {
          document.removeEventListener("mousemove", handleMouseMove);
          document.removeEventListener("mouseup", handleMouseUp);
        };
      }
    }, [isDragging, selectedValue]);

    if (!options) {
      return null;
    }

    return (
      <div className="space-y-2" ref={ref}>
        {label && <label className="text-sm font-medium">{label}</label>}
        <div className="relative">
          <div className="absolute top-[9px] left-2 right-2 h-1 bg-neutral-200 dark:bg-neutral-700 pointer-events-none rounded-full" />

          <div className="flex justify-between" ref={containerRef}>
            {options.map((option) => (
              <div key={option.value} className="flex flex-col items-center">
                <button
                  onClick={() => handleClick(option.value)}
                  onMouseDown={handleMouseDown}
                  className="relative px-3 py-6 -mx-3 -my-6 z-10 cursor-pointer"
                >
                  <div className="relative w-5 h-5 flex items-center justify-center">
                    {selectedValue === option.value && (
                      <div className="w-4 h-4 bg-white dark:bg-white border border-neutral-400 dark:border-neutral-500 rounded-full cursor-grab active:cursor-grabbing" />
                    )}
                  </div>
                </button>
                <div className="text-xs mt text-neutral-500 dark:text-neutral-400">
                  {option.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  },
);

Slider.displayName = "Slider";

export { Slider };
