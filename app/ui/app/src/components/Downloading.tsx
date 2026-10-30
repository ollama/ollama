const K = 1024;
const SIZES = ["B", "KB", "MB", "GB", "TB"];

function formatBytes(bytes: number, unit?: string): string {
  let i: number;
  if (unit) {
    i = SIZES.indexOf(unit);
  } else {
    i = bytes === 0 ? 0 : Math.floor(Math.log(bytes) / Math.log(K));
  }

  const decimals = SIZES[i] === "GB" || SIZES[i] === "TB" ? 1 : 0;
  return `${(bytes / Math.pow(K, i)).toFixed(decimals)} ${SIZES[i]}`;
}

export default function Downloading({
  completed,
  total,
}: {
  completed: number;
  total: number;
}) {
  const percentage = total > 0 ? (completed / total) * 100 : 0;
  const unitIndex = total > 0 ? Math.floor(Math.log(total) / Math.log(K)) : 0;
  const unit = SIZES[unitIndex];

  return (
    <div className="my-4 rounded-xl max-w-xs">
      <div className="flex flex-col mb-2 text-neutral-800 dark:text-neutral-200">
        <div className="flex items-center mb-0.5">
          <svg
            className="w-3.5 absolute fill-current"
            viewBox="0 0 18 25"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M17.334 11.1133V18.6328C17.334 21.25 15.8691 22.7148 13.2422 22.7148H4.08203C1.45508 22.7148 0 21.25 0 18.6328V11.1133C0 8.48633 1.45508 7.03125 4.08203 7.03125H6.05469V8.60352H4.08203C2.48047 8.60352 1.57227 9.51172 1.57227 11.1133V18.6328C1.57227 20.2344 2.48047 21.1426 4.08203 21.1426H13.2422C14.8535 21.1426 15.7617 20.2344 15.7617 18.6328V11.1133C15.7617 9.51172 14.8535 8.60352 13.2422 8.60352H11.2793V7.03125H13.2422C15.8691 7.03125 17.334 8.48633 17.334 11.1133Z" />
            <path d="M8.67188 1.83594C8.25195 1.83594 7.89062 2.17773 7.89062 2.58789V12.5195L8.00781 15.1562C8.02734 15.5176 8.31055 15.8105 8.67188 15.8105C9.02344 15.8105 9.30664 15.5176 9.32617 15.1562L9.44336 12.5195V2.58789C9.44336 2.17773 9.0918 1.83594 8.67188 1.83594ZM5.35156 11.5625C4.94141 11.5625 4.64844 11.8457 4.64844 12.2461C4.64844 12.4609 4.73633 12.6172 4.88281 12.7637L8.10547 15.8691C8.30078 16.0645 8.4668 16.123 8.67188 16.123C8.86719 16.123 9.0332 16.0645 9.22852 15.8691L12.4512 12.7637C12.5977 12.6172 12.6855 12.4609 12.6855 12.2461C12.6855 11.8457 12.373 11.5625 11.9727 11.5625C11.7773 11.5625 11.582 11.6406 11.4453 11.7969L9.93164 13.4082L8.67188 14.7461L7.40234 13.4082L5.88867 11.7969C5.75195 11.6406 5.54688 11.5625 5.35156 11.5625Z" />
          </svg>

          <div className="ml-6">Downloading model</div>
        </div>
        <div className="text-sm text-neutral-500 dark:text-neutral-500 ml-6">
          {`${formatBytes(completed, unit)} / ${formatBytes(total, unit)} (${Math.floor(percentage)}%)`}
        </div>
      </div>
      <div className="relative h-1.5 bg-neutral-200 dark:bg-neutral-700 rounded-full overflow-hidden ml-6">
        <div
          className="absolute left-0 top-0 h-full bg-neutral-700 dark:bg-neutral-500 rounded-full"
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>
    </div>
  );
}
