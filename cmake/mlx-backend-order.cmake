function(ollama_order_mlx_metal_builds)
    if(TARGET ollama-mlx-metal_v3 AND TARGET ollama-mlx-metal_v4)
        # Both external projects install shared notices and licenses into the
        # same payload root. Keep their complete install phases from racing.
        add_dependencies(ollama-mlx-metal_v4 ollama-mlx-metal_v3)
    endif()
endfunction()
