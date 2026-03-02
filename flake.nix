{
  description = "PaddleOCR plugin for OCRmyPDF";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [(final: prev: {
          python312 = prev.python312.override {
            packageOverrides = _pyfinal: pyprev: {
              # behave 1.3.3 has a flaky test failure in nixpkgs master; disable checks
              # (behave is only a test dep of python-docx, which is pulled in by pdf2docx/paddlex)
              behave = pyprev.behave.overridePythonAttrs (_: { doCheck = false; });
            };
          };
        })];
      };

      python = pkgs.python312;
      pythonPackages = python.pkgs;

      # PaddleX OCR extra dependencies (nixpkgs paddlex has no optional-dependencies)
      paddlex-ocr-deps = with pythonPackages; [
        einops
        ftfy
        imagesize
        jinja2
        lxml
        openpyxl
        pyclipper
        pypdfium2
        regex
        scikit-learn
        shapely
        tiktoken
        tokenizers
        python-bidi
        sentencepiece
        # opencv-contrib-python and premailer handled by shims
      ];

      # Override paddlepaddle to skip strict wheel METADATA version checks.
      # The wheel pins opt-einsum==3.3.0 but nixpkgs ships 3.4.0 (compatible);
      # networkx is only used for paddle graph features, not OCR.
      paddlepaddle-relaxed = pythonPackages.paddlepaddle.overridePythonAttrs (old: {
        # dontCheckRuntimeDeps is read by pythonRuntimeDepsCheckHook as a shell var
        env.dontCheckRuntimeDeps = "1";
        # Fix numpy 2.x incompatibility: int(np.array(multi_element)) raises TypeError
        # in numpy 2.x when the array has >0 dimensions. Use .item() for scalar extraction.
        # Affects paddle/base/dygraph/math_op_patch.py __int__ / __index__ methods.
        postInstall = (old.postInstall or "") + ''
          substituteInPlace "$out/${python.sitePackages}/paddle/base/dygraph/math_op_patch.py" \
            --replace \
              'return int(np.array(var))' \
              'arr = np.array(var); return int(arr.item() if arr.ndim > 0 else arr)'
        '';
      });

      # Override paddlex to include OCR dependencies
      paddlex-with-ocr = pythonPackages.paddlex.overridePythonAttrs (old: {
        dependencies = (old.dependencies or []) ++ paddlex-ocr-deps;
      });

      # Override paddleocr to use paddlex-with-ocr and paddlepaddle-relaxed
      paddleocr-with-ocr = (pythonPackages.paddleocr.override {
        paddlex = paddlex-with-ocr;
        paddlepaddle = paddlepaddle-relaxed;
      });

      # PaddleX compatibility shims (opencv-contrib-python and premailer)
      python-shims = pkgs.runCommand "paddlex-compat-shims" {} ''
        mkdir -p $out/${python.sitePackages}

        cat > $out/${python.sitePackages}/opencv_contrib_python.py << 'PYEOF'
# Shim: PaddleX checks for opencv-contrib-python; nixpkgs provides opencv4
from cv2 import *
PYEOF

        cat > $out/${python.sitePackages}/premailer.py << 'PYEOF'
# Shim: PaddleX checks for premailer (only used for HTML->Excel, not OCR)
class Premailer:
    pass
PYEOF

        mkdir -p $out/${python.sitePackages}/opencv_contrib_python-4.10.0.84.dist-info
        cat > $out/${python.sitePackages}/opencv_contrib_python-4.10.0.84.dist-info/METADATA << 'EOF'
Metadata-Version: 2.1
Name: opencv-contrib-python
Version: 4.10.0.84
EOF

        mkdir -p $out/${python.sitePackages}/premailer-3.10.0.dist-info
        cat > $out/${python.sitePackages}/premailer-3.10.0.dist-info/METADATA << 'EOF'
Metadata-Version: 2.1
Name: premailer
Version: 3.10.0
EOF
      '';

      # The plugin package
      ocrmypdf-paddleocr = pythonPackages.buildPythonPackage {
        pname = "ocrmypdf-paddleocr";
        version = "0.1.0";
        pyproject = true;

        src = builtins.path {
          path = ./.;
          name = "ocrmypdf-paddleocr-src";
          filter = path: type:
            let baseName = builtins.baseNameOf path;
            in !(
              baseName == "OCRmyPDF" ||
              baseName == "PaddleOCR" ||
              baseName == "PaddleX" ||
              baseName == ".git" ||
              baseName == ".python-shims" ||
              baseName == "result" ||
              baseName == "flake.nix" ||
              baseName == "flake.lock" ||
              baseName == "shell.nix" ||
              baseName == "default.nix" ||
              (type == "regular" && pkgs.lib.hasSuffix ".pdf" baseName)
            );
        };

        env.SETUPTOOLS_SCM_PRETEND_VERSION = "0.1.0";

        build-system = with pythonPackages; [
          setuptools
          setuptools-scm
          wheel
        ];

        dependencies = with pythonPackages; [
          (ocrmypdf.override {
            img2pdf = img2pdf.overridePythonAttrs (old: { doCheck = false; });
          })
          paddlepaddle-relaxed
          paddleocr-with-ocr
          pillow
        ];

        doCheck = false;

        pythonImportsCheck = [ "ocrmypdf_paddleocr" ];

        meta = {
          description = "PaddleOCR plugin for OCRmyPDF";
          license = pkgs.lib.licenses.mpl20;
          platforms = pkgs.lib.platforms.linux;
        };
      };

      # Python environment with all runtime dependencies
      pythonEnv = python.withPackages (_ps: [
        ocrmypdf-paddleocr
        paddlex-with-ocr
        paddleocr-with-ocr
      ]);

      shimPath = "${python-shims}/${python.sitePackages}";

      # Wrapped ocrmypdf binary with plugin pre-loaded
      ocrmypdf-wrapped = pkgs.writeShellScriptBin "ocrmypdf" ''
        export PYTHONPATH="${shimPath}''${PYTHONPATH:+:$PYTHONPATH}"
        export PATH="${pkgs.lib.makeBinPath [ pkgs.ghostscript pkgs.pngquant pkgs.unpaper ]}''${PATH:+:$PATH}"
        # Work around PaddlePaddle OneDNN PIR bug (ConvertPirAttribute2RuntimeAttribute)
        export PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=''${PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT:-0}
        exec ${pythonEnv}/bin/ocrmypdf --plugin ocrmypdf_paddleocr "$@"
      '';

    in {
      packages.${system} = {
        default = ocrmypdf-wrapped;
        plugin = ocrmypdf-paddleocr;
      };

      apps.${system}.default = {
        type = "app";
        program = "${ocrmypdf-wrapped}/bin/ocrmypdf";
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          (python.withPackages (_ps: [
            ocrmypdf-paddleocr
            paddlex-with-ocr
            paddleocr-with-ocr
            pythonPackages.python-bidi
            pythonPackages.sentencepiece
            pythonPackages.pytest
            pythonPackages.ipython
          ]))
          pkgs.ghostscript
          pkgs.pngquant
          pkgs.unpaper
        ];

        shellHook = ''
          export PYTHONPATH="${shimPath}''${PYTHONPATH:+:$PYTHONPATH}"

          echo "OCRmyPDF-PaddleOCR development environment"
          echo "========================================="
          echo ""
          echo "Python: $(python --version)"
          echo "OCRmyPDF: $(python -c 'import ocrmypdf; print(ocrmypdf.__version__)' 2>/dev/null || echo 'not found')"
          echo "PaddleOCR: $(python -c 'import paddleocr; print(paddleocr.__version__)' 2>/dev/null || echo 'not found')"
          echo ""
          echo "Usage:"
          echo "  ocrmypdf --plugin ocrmypdf_paddleocr input.pdf output.pdf"
          echo ""
        '';
      };
    };
}
