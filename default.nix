with (import <nixpkgs> {});
let
  LLP = with pkgs; [
    gcc11
    cudatoolkit
    linuxPackages.nvidia_x11
    go
    cmake
  ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath LLP;
in  
stdenv.mkDerivation {
  name = "ollama-env";
  buildInputs = LLP;
  src = null;
  # IMPORTANT: Edit ./llm/generate/gen_linux.sh
  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export CUDA_LIB_DIR=${cudatoolkit.out}/lib
    export CUDART_LIB_DIR=${cudatoolkit.lib}/lib
    export NVCC_PREPEND_FLAGS='-ccbin ${gcc11}/bin/'
  '';
}
