#!/usr/bin/env bash

# Vendored from https://github.com/create-dmg/create-dmg so our build can be self-contained

# Create a read-only disk image of the contents of a folder

# Bail out on any unhandled errors
set -e;
# Any command that exits with non-zero code will cause the pipeline to fail
set -o pipefail;

CDMG_VERSION='1.2.1'

# The full path to the "support/" directory this script is using
# (This will be set up by code later in the script.)
CDMG_SUPPORT_DIR=""

OS_FULL_VERSION="$(sw_vers | sed -n 2p | cut -d : -f 2 | tr -d '[:space:]' | cut -c1-)"
OS_MAJOR_VERSION="$(echo $OS_FULL_VERSION | cut -d . -f 1)"
OS_MINOR_VERSION="$(echo $OS_FULL_VERSION | cut -d . -f 2)"
WINX=10
WINY=60
WINW=500
WINH=350
ICON_SIZE=128
TEXT_SIZE=16
FORMAT="UDZO"
FILESYSTEM="HFS+"
ADD_FILE_SOURCES=()
ADD_FILE_TARGETS=()
IMAGEKEY=""
HDIUTIL_VERBOSITY=""
SANDBOX_SAFE=0
BLESS=0
SKIP_JENKINS=0
MAXIMUM_UNMOUNTING_ATTEMPTS=3
SIGNATURE=""
NOTARIZE=""

function pure_version() {
	echo "$CDMG_VERSION"
}

function hdiutil_detach_retry() {
	# Unmount
	unmounting_attempts=0
	until
		echo "Unmounting disk image..."
		(( unmounting_attempts++ ))
		hdiutil detach "$1"
		exit_code=$?
		(( exit_code ==  0 )) && break            # nothing goes wrong
		(( exit_code != 16 )) && exit $exit_code  # exit with the original exit code
		# The above statement returns 1 if test failed (exit_code == 16).
		#   It can make the code in the {do... done} block to be executed
	do
		(( unmounting_attempts == MAXIMUM_UNMOUNTING_ATTEMPTS )) && exit 16  # patience exhausted, exit with code EBUSY
		echo "Wait a moment..."
		sleep $(( 1 * (2 ** unmounting_attempts) ))
	done
	unset unmounting_attempts
}

function version() {
	echo "create-dmg $(pure_version)"
}

function usage() {
	version
	cat <<EOHELP

Creates a fancy DMG file.

Usage:  $(basename $0) [options] <output_name.dmg> <source_folder>

All contents of <source_folder> will be copied into the disk image.

Options:
  --volname <name>
      set volume name (displayed in the Finder sidebar and window title)
  --volicon <icon.icns>
      set volume icon
  --background <pic.png>
      set folder background image (provide png, gif, or jpg)
  --window-pos <x> <y>
      set position the folder window
  --window-size <width> <height>
      set size of the folder window
  --text-size <text_size>
      set window text size (10-16)
  --icon-size <icon_size>
      set window icons size (up to 128)
  --icon file_name <x> <y>
      set position of the file's icon
  --hide-extension <file_name>
      hide the extension of file
  --app-drop-link <x> <y>
      make a drop link to Applications, at location x,y
  --ql-drop-link <x> <y>
      make a drop link to user QuickLook install dir, at location x,y
  --eula <eula_file>
      attach a license file to the dmg (plain text or RTF)
  --no-internet-enable
      disable automatic mount & copy
  --format <format>
      specify the final disk image format (UDZO|UDBZ|ULFO|ULMO) (default is UDZO)
  --filesystem <filesystem>
      specify the disk image filesystem (HFS+|APFS) (default is HFS+, APFS supports macOS 10.13 or newer)
  --encrypt
      enable encryption for the resulting disk image (AES-256 - you will be prompted for password)
  --encrypt-aes128
      enable encryption for the resulting disk image (AES-128 - you will be prompted for password)
  --add-file <target_name> <file>|<folder> <x> <y>
      add additional file or folder (can be used multiple times)
  --disk-image-size <x>
      set the disk image size manually to x MB
  --hdiutil-verbose
      execute hdiutil in verbose mode
  --hdiutil-quiet
      execute hdiutil in quiet mode
  --bless
      bless the mount folder (deprecated, needs macOS 12.2.1 or older)
  --codesign <signature>
      codesign the disk image with the specified signature
  --notarize <credentials>
      notarize the disk image (waits and staples) with the keychain stored credentials
  --sandbox-safe
      execute hdiutil with sandbox compatibility and do not bless (not supported for APFS disk images)
  --skip-jenkins
      skip Finder-prettifying AppleScript, useful in Sandbox and non-GUI environments
  --version
	    show create-dmg version number
  -h, --help
	    display this help screen

EOHELP
	exit 0
}

# factors can cause interstitial disk images to contain more than a single
# partition - expand the hunt for the temporary disk image by checking for
# the path of the volume, versus assuming its the first result (as in pr/152).
function find_mount_dir() {
	local dev_name="${1}"

	>&2 echo "Searching for mounted interstitial disk image using ${dev_name}... "
	# enumerate up to 9 partitions
	for i in {1..9}; do
		# attempt to find the partition
		local found_dir
		found_dir=$(hdiutil info | grep -E --color=never "${dev_name}" | head -${i} | awk '{print $3}' | xargs)
		if [[ -n "${found_dir}" ]]; then
				echo "${found_dir}"
				return 0
		fi
	done
}

# Argument parsing

while [[ "${1:0:1}" = "-" ]]; do
	case $1 in
		--volname)
			VOLUME_NAME="$2"
			shift; shift;;
		--volicon)
			VOLUME_ICON_FILE="$2"
			shift; shift;;
		--background)
			BACKGROUND_FILE="$2"
			BACKGROUND_FILE_NAME="$(basename "$BACKGROUND_FILE")"
			BACKGROUND_CLAUSE="set background picture of opts to file \".background:$BACKGROUND_FILE_NAME\""
			REPOSITION_HIDDEN_FILES_CLAUSE="set position of every item to {theBottomRightX + 100, 100}"
			shift; shift;;
		--icon-size)
			ICON_SIZE="$2"
			shift; shift;;
		--text-size)
			TEXT_SIZE="$2"
			shift; shift;;
		--window-pos)
			WINX=$2; WINY=$3
			shift; shift; shift;;
		--window-size)
			WINW=$2; WINH=$3
			shift; shift; shift;;
		--icon)
			POSITION_CLAUSE="${POSITION_CLAUSE}set position of item \"$2\" to {$3, $4}
			"
			shift; shift; shift; shift;;
		--hide-extension)
			HIDING_CLAUSE="${HIDING_CLAUSE}set the extension hidden of item \"$2\" to true
			"
			shift; shift;;
		-h | --help)
			usage;;
		--version)
			version; exit 0;;
		--pure-version)
			pure_version; exit 0;;
		--ql-drop-link)
			QL_LINK=$2
			QL_CLAUSE="set position of item \"QuickLook\" to {$2, $3}
			"
			shift; shift; shift;;
		--app-drop-link)
			APPLICATION_LINK=$2
			APPLICATION_CLAUSE="set position of item \"Applications\" to {$2, $3}
			"
			shift; shift; shift;;
		--eula)
			EULA_RSRC=$2
			shift; shift;;
		--no-internet-enable)
			NOINTERNET=1
			shift;;
		--format)
			FORMAT="$2"
			shift; shift;;
		--filesystem)
			FILESYSTEM="$2"
			shift; shift;;
		--encrypt)
			ENABLE_ENCRYPTION=1
			AESBITS=256
			shift;;
		--encrypt-aes128)
			ENABLE_ENCRYPTION=1
			AESBITS=128
			shift;;
		--add-file | --add-folder)
			ADD_FILE_TARGETS+=("$2")
			ADD_FILE_SOURCES+=("$3")
			POSITION_CLAUSE="${POSITION_CLAUSE}
			set position of item \"$2\" to {$4, $5}
			"
			shift; shift; shift; shift; shift;;
		--disk-image-size)
			DISK_IMAGE_SIZE="$2"
			shift; shift;;
		--hdiutil-verbose)
			HDIUTIL_VERBOSITY='-verbose'
			shift;;
		--hdiutil-quiet)
			HDIUTIL_VERBOSITY='-quiet'
			shift;;
		--codesign)
			SIGNATURE="$2"
			shift; shift;;
		--notarize)
			NOTARIZE="$2"
			shift; shift;;
		--sandbox-safe)
			SANDBOX_SAFE=1
			shift;;
		--bless)
			BLESS=1
			shift;;
		--rez)
			echo "REZ is no more directly used. You can remove the --rez argument."
			shift; shift;;
		--skip-jenkins)
			SKIP_JENKINS=1
			shift;;
		-*)
			echo "Unknown option: $1. Run 'create-dmg --help' for help."
			exit 1;;
	esac
	case $FORMAT in
		UDZO)
			IMAGEKEY="-imagekey zlib-level=9";;
		UDBZ)
			IMAGEKEY="-imagekey bzip2-level=9";;
		ULFO)
			;;
		ULMO)
			;;
		*)
			echo >&2 "Unknown disk image format: $FORMAT"
			exit 1;;
	esac
done

if [[ -z "$2" ]]; then
	echo "Not enough arguments. Run 'create-dmg --help' for help."
	exit 1
fi

DMG_PATH="$1"
SRC_FOLDER="$(cd "$2" > /dev/null; pwd)"

# Argument validation checks

if [[ "${DMG_PATH: -4}" != ".dmg" ]]; then
	echo "Output file name must end with a .dmg extension. Run 'create-dmg --help' for help."
	exit 1
fi

if [[ "${FILESYSTEM}" != "HFS+" ]] && [[ "${FILESYSTEM}" != "APFS" ]]; then
	echo "Unknown disk image filesystem: ${FILESYSTEM}. Run 'create-dmg --help' for help."
	exit 1
fi

if [[ "${FILESYSTEM}" == "APFS" ]] && [[ ${SANDBOX_SAFE} -eq 1 ]]; then
	echo "Creating an APFS disk image that is sandbox safe is not supported."
	exit 1
fi

# Main script logic

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DMG_DIRNAME="$(dirname "$DMG_PATH")"
DMG_DIR="$(cd "$DMG_DIRNAME" > /dev/null; pwd)"
DMG_NAME="$(basename "$DMG_PATH")"
DMG_TEMP_NAME="$DMG_DIR/rw.$$.${DMG_NAME}"

# Detect where we're running from

sentinel_file="$SCRIPT_DIR/.this-is-the-create-dmg-repo"
if [[ -f "$sentinel_file" ]]; then
	# We're running from inside a repo
	CDMG_SUPPORT_DIR="$SCRIPT_DIR/support"
else
	# We're running inside an installed location
	bin_dir="$SCRIPT_DIR"
	prefix_dir=$(dirname "$bin_dir")
	CDMG_SUPPORT_DIR="$prefix_dir/share/create-dmg/support"
fi

if [[ -z "$VOLUME_NAME" ]]; then
	VOLUME_NAME="$(basename "$DMG_PATH" .dmg)"
fi

if [[ ! -d "$CDMG_SUPPORT_DIR" ]]; then
	echo >&2 "Cannot find support/ directory: expected at: $CDMG_SUPPORT_DIR"
	exit 1
fi

if [[ -f "$SRC_FOLDER/.DS_Store" ]]; then
	echo "Deleting .DS_Store found in source folder"
	rm "$SRC_FOLDER/.DS_Store"
fi

# Create the image
echo "Creating disk image..."
if [[ -f "${DMG_TEMP_NAME}" ]]; then
	rm -f "${DMG_TEMP_NAME}"
fi

# Use Megabytes since hdiutil fails with very large byte numbers
function blocks_to_megabytes() {
	# Add 1 extra MB, since there's no decimal retention here
	MB_SIZE=$((($1 * 512 / 1000 / 1000) + 1))
	echo $MB_SIZE
}

function get_size() {
	# Get block size in disk
	if [[ $OS_MAJOR_VERSION -ge 12 ]]; then
		bytes_size=$(du -B 512 -s "$1")
	else
		bytes_size=$(du -s "$1")
	fi
	bytes_size=$(echo $bytes_size | sed -e 's/	.*//g')
	echo $(blocks_to_megabytes $bytes_size)
}

# Create the DMG with the specified size or the hdiutil estimation
CUSTOM_SIZE=''
if [[ -n "$DISK_IMAGE_SIZE" ]]; then
	CUSTOM_SIZE="-size ${DISK_IMAGE_SIZE}m"
fi

if [[ $SANDBOX_SAFE -eq 0 ]]; then
	if [[ "$FILESYSTEM" == "APFS" ]]; then
		FILESYSTEM_ARGUMENTS=""
	else
		FILESYSTEM_ARGUMENTS="-c c=64,a=16,e=16"
	fi
	hdiutil create ${HDIUTIL_VERBOSITY} -srcfolder "$SRC_FOLDER" -volname "${VOLUME_NAME}" \
		-fs "${FILESYSTEM}" -fsargs "${FILESYSTEM_ARGUMENTS}" -format UDRW ${CUSTOM_SIZE} "${DMG_TEMP_NAME}"
else
	hdiutil makehybrid ${HDIUTIL_VERBOSITY} -default-volume-name "${VOLUME_NAME}" -hfs -o "${DMG_TEMP_NAME}" "$SRC_FOLDER"
	hdiutil convert -format UDRW -ov -o "${DMG_TEMP_NAME}" "${DMG_TEMP_NAME}"
	DISK_IMAGE_SIZE_CUSTOM=$DISK_IMAGE_SIZE
fi

# Get the created DMG actual size
DISK_IMAGE_SIZE=$(get_size "${DMG_TEMP_NAME}")

# Use the custom size if bigger
if [[ $SANDBOX_SAFE -eq 1 ]] && [[ ! -z "$DISK_IMAGE_SIZE_CUSTOM" ]] && [[ $DISK_IMAGE_SIZE_CUSTOM -gt $DISK_IMAGE_SIZE ]]; then
	DISK_IMAGE_SIZE=$DISK_IMAGE_SIZE_CUSTOM
fi

# Estimate the additional sources size
if [[ -n "$ADD_FILE_SOURCES" ]]; then
	for i in "${!ADD_FILE_SOURCES[@]}"; do
		SOURCE_SIZE=$(get_size "${ADD_FILE_SOURCES[$i]}")
		DISK_IMAGE_SIZE=$(expr $DISK_IMAGE_SIZE + $SOURCE_SIZE)
	done
fi

# Add extra space for additional resources
DISK_IMAGE_SIZE=$(expr $DISK_IMAGE_SIZE + 20)

# Make sure target image size is within limits
MIN_DISK_IMAGE_SIZE=$(hdiutil resize -limits "${DMG_TEMP_NAME}" | awk 'NR=1{print int($1/2048+1)}')
if [ $MIN_DISK_IMAGE_SIZE -gt $DISK_IMAGE_SIZE ]; then
       DISK_IMAGE_SIZE=$MIN_DISK_IMAGE_SIZE
fi

# Resize the image for the extra stuff
hdiutil resize ${HDIUTIL_VERBOSITY} -size ${DISK_IMAGE_SIZE}m "${DMG_TEMP_NAME}"

# Mount the new DMG

echo "Mounting disk image..."

MOUNT_RANDOM_PATH="/Volumes"
if [[ $SANDBOX_SAFE -eq 1 ]]; then
	MOUNT_RANDOM_PATH="/tmp"
fi
if [[ "$FILESYSTEM" == "APFS" ]]; then
  HDIUTIL_FILTER="tail -n 1"
else
  HDIUTIL_FILTER="sed 1q"
fi
DEV_NAME=$(hdiutil attach -mountrandom ${MOUNT_RANDOM_PATH} -readwrite -noverify -noautoopen -nobrowse "${DMG_TEMP_NAME}" | grep -E --color=never '^/dev/' | ${HDIUTIL_FILTER} | awk '{print $1}')
echo "Device name:     $DEV_NAME"
if [[ "$FILESYSTEM" == "APFS" ]]; then
  MOUNT_DIR=$(find_mount_dir "${DEV_NAME}")
else
	MOUNT_DIR=$(find_mount_dir "${DEV_NAME}s")
fi
if [[ -z "${MOUNT_DIR}" ]]; then
  >&2 echo "ERROR: unable to proceed with final disk image creation because the interstitial disk image was not found."
  >&2 echo "The interstitial disk image will likely be mounted and will need to be cleaned up manually."
  exit 1
fi

echo "Mount dir:       $MOUNT_DIR"

if [[ -n "$BACKGROUND_FILE" ]]; then
	echo "Copying background file '$BACKGROUND_FILE'..."
	[[ -d "$MOUNT_DIR/.background" ]] || mkdir "$MOUNT_DIR/.background"
	cp "$BACKGROUND_FILE" "$MOUNT_DIR/.background/$BACKGROUND_FILE_NAME"
fi

if [[ -n "$APPLICATION_LINK" ]]; then
	echo "Making link to Applications dir..."
	echo $MOUNT_DIR
	ln -s /Applications "$MOUNT_DIR/Applications"
fi

if [[ -n "$QL_LINK" ]]; then
	echo "Making link to QuickLook install dir..."
	echo $MOUNT_DIR
	ln -s "/Library/QuickLook" "$MOUNT_DIR/QuickLook"
fi

if [[ -n "$VOLUME_ICON_FILE" ]]; then
	echo "Copying volume icon file '$VOLUME_ICON_FILE'..."
	cp "$VOLUME_ICON_FILE" "$MOUNT_DIR/.VolumeIcon.icns"
	SetFile -c icnC "$MOUNT_DIR/.VolumeIcon.icns"
fi

if [[ -n "$ADD_FILE_SOURCES" ]]; then
	echo "Copying custom files..."
	for i in "${!ADD_FILE_SOURCES[@]}"; do
		echo "${ADD_FILE_SOURCES[$i]}"
		cp -a "${ADD_FILE_SOURCES[$i]}" "$MOUNT_DIR/${ADD_FILE_TARGETS[$i]}"
	done
fi

VOLUME_NAME=$(basename $MOUNT_DIR)

# Run AppleScript to do all the Finder cosmetic stuff
APPLESCRIPT_FILE=$(mktemp -t createdmg.tmp.XXXXXXXXXX)
if [[ $SANDBOX_SAFE -eq 1 ]]; then
	echo "Skipping Finder-prettifying AppleScript because we are in Sandbox..."
else
	if [[ $SKIP_JENKINS -eq 0 ]]; then
		cat "$CDMG_SUPPORT_DIR/template.applescript" \
			| sed -e "s/WINX/$WINX/g" -e "s/WINY/$WINY/g" -e "s/WINW/$WINW/g" \
					-e "s/WINH/$WINH/g" -e "s/BACKGROUND_CLAUSE/$BACKGROUND_CLAUSE/g" \
					-e "s/REPOSITION_HIDDEN_FILES_CLAUSE/$REPOSITION_HIDDEN_FILES_CLAUSE/g" \
					-e "s/ICON_SIZE/$ICON_SIZE/g" -e "s/TEXT_SIZE/$TEXT_SIZE/g" \
			| perl -pe "s/POSITION_CLAUSE/$POSITION_CLAUSE/g" \
			| perl -pe "s/QL_CLAUSE/$QL_CLAUSE/g" \
			| perl -pe "s/APPLICATION_CLAUSE/$APPLICATION_CLAUSE/g" \
			| perl -pe "s/HIDING_CLAUSE/$HIDING_CLAUSE/" \
			> "$APPLESCRIPT_FILE"

		# pause to workaround occasional "Can’t get disk" (-1728) issues
		ERROR_1728_WORKAROUND_SLEEP_INTERVAL=2
		echo "Will sleep for $ERROR_1728_WORKAROUND_SLEEP_INTERVAL seconds to workaround occasions \"Can't get disk (-1728)\" issues..."
		sleep $ERROR_1728_WORKAROUND_SLEEP_INTERVAL

		echo "Running AppleScript to make Finder stuff pretty: /usr/bin/osascript \"${APPLESCRIPT_FILE}\" \"${VOLUME_NAME}\""
		if /usr/bin/osascript "${APPLESCRIPT_FILE}" "${VOLUME_NAME}"; then
			# Okay, we're cool
			true
		else
			echo >&2 "Failed running AppleScript"
			hdiutil_detach_retry "${DEV_NAME}"
			exit 64
		fi
		echo "Done running the AppleScript..."
		sleep 4
		rm "$APPLESCRIPT_FILE"
	else
		echo ''
		echo "Will skip running AppleScript to configure DMG aesthetics because of --skip-jenkins option."
		echo "This will result in a DMG without any custom background or icons positioning."
		echo "More info at https://github.com/create-dmg/create-dmg/issues/72"
		echo ''
	fi
fi

# Make sure it's not world writeable
echo "Fixing permissions..."
chmod -Rf go-w "${MOUNT_DIR}" &> /dev/null || true
echo "Done fixing permissions"

# Make the top window open itself on mount:
if [[ $BLESS -eq 1 && $SANDBOX_SAFE -eq 0 ]]; then
	echo "Blessing started"
	if [ $(uname -m) == "arm64" ]; then
		bless --folder "${MOUNT_DIR}"
	else
		bless --folder "${MOUNT_DIR}" --openfolder "${MOUNT_DIR}"
	fi
	echo "Blessing finished"
else
	echo "Skipping blessing on sandbox"
fi

if [[ -n "$VOLUME_ICON_FILE" ]]; then
	# Tell the volume that it has a special file attribute
	SetFile -a C "$MOUNT_DIR"
fi

# Delete unnecessary file system events log if possible
echo "Deleting .fseventsd"
rm -rf "${MOUNT_DIR}/.fseventsd" || true

hdiutil_detach_retry "${DEV_NAME}"

# Compress image and optionally encrypt
if [[ $ENABLE_ENCRYPTION -eq 0 ]]; then
	echo "Compressing disk image..."
	hdiutil convert ${HDIUTIL_VERBOSITY} "${DMG_TEMP_NAME}" -format ${FORMAT} ${IMAGEKEY} -o "${DMG_DIR}/${DMG_NAME}"
else
	echo "Compressing and encrypting disk image..."
	echo "NOTE: hdiutil will only prompt a single time for a password - ensure entry is correct."
	hdiutil convert ${HDIUTIL_VERBOSITY} "${DMG_TEMP_NAME}" -format ${FORMAT} ${IMAGEKEY} -encryption AES-${AESBITS} -stdinpass -o "${DMG_DIR}/${DMG_NAME}"
fi
rm -f "${DMG_TEMP_NAME}"

# Adding EULA resources
if [[ -n "${EULA_RSRC}" && "${EULA_RSRC}" != "-null-" ]]; then
	echo "Adding EULA resources..."
	#
	# Use udifrez instead flatten/rez/unflatten
	# https://github.com/create-dmg/create-dmg/issues/109
	#
	# Based on a thread from dawn2dusk & peterguy
	# https://developer.apple.com/forums/thread/668084
	#
	EULA_RESOURCES_FILE=$(mktemp -t createdmg.tmp.XXXXXXXXXX)
	EULA_FORMAT=$(file -b ${EULA_RSRC})
	if [[ ${EULA_FORMAT} == 'Rich Text Format data'* ]] ; then
		EULA_FORMAT='RTF '
	else
		EULA_FORMAT='TEXT'
	fi
	# Encode the EULA to base64
	# Replace 'openssl base64' with 'base64' if Mac OS X 10.6 support is no more needed
	# EULA_DATA="$(base64 -b 52 "${EULA_RSRC}" | sed s$'/^\(.*\)$/\t\t\t\\1/')"
	EULA_DATA="$(openssl base64 -in "${EULA_RSRC}" | tr -d '\n' | awk '{gsub(/.{52}/,"&\n")}1' | sed s$'/^\(.*\)$/\t\t\t\\1/')"
	# Fill the template with the custom EULA contents
	eval "cat > \"${EULA_RESOURCES_FILE}\" <<EOF
	$(<${CDMG_SUPPORT_DIR}/eula-resources-template.xml)
	EOF
	"
	# Apply the resources
	hdiutil udifrez -xml "${EULA_RESOURCES_FILE}" '' -quiet "${DMG_DIR}/${DMG_NAME}" || {
		echo "Failed to add the EULA license"
		exit 1
	}
	echo "Successfully added the EULA license"
fi

# Enable "internet", whatever that is
if [[ ! -z "${NOINTERNET}" && "${NOINTERNET}" == 1 ]]; then
	echo "Not setting 'internet-enable' on the dmg, per caller request"
else
	# Check if hdiutil supports internet-enable
	# Support was removed in macOS 10.15. See https://github.com/andreyvit/create-dmg/issues/76
	if hdiutil internet-enable -help >/dev/null 2>/dev/null; then
		hdiutil internet-enable -yes "${DMG_DIR}/${DMG_NAME}"
	else
		echo "hdiutil does not support internet-enable. Note it was removed in macOS 10.15."
	fi
fi

if [[ -n "${SIGNATURE}" && "${SIGNATURE}" != "-null-" ]]; then
	echo "Codesign started"
	codesign -s "${SIGNATURE}" "${DMG_DIR}/${DMG_NAME}"
	dmgsignaturecheck="$(codesign --verify --deep --verbose=2 --strict "${DMG_DIR}/${DMG_NAME}" 2>&1 >/dev/null)"
	if [ $? -eq 0 ]; then
		echo "The disk image is now codesigned"
	else
		echo "The signature seems invalid${NC}"
		exit 1
	fi
fi

if [[ -n "${NOTARIZE}" && "${NOTARIZE}" != "-null-" ]]; then
	echo "Notarization started"
	xcrun notarytool submit "${DMG_DIR}/${DMG_NAME}" --keychain-profile "${NOTARIZE}" --wait
	echo "Stapling the notarization ticket"
	staple="$(xcrun stapler staple "${DMG_DIR}/${DMG_NAME}")"
	if [ $? -eq 0 ]; then
		echo "The disk image is now notarized"
	else
		echo "$staple"
		echo "The notarization failed with error $?"
		exit 1
	fi
fi

# All done!
echo "Disk image done"
exit 0
