# SPDX-FileCopyrightText: 2025 IObundle
#
# SPDX-License-Identifier: MIT

{ pkgs ? import <nixpkgs> {} }:

let
  py2hwsw_commit = "b4ddec01e54954573b0a50d581411f34d0fec8dc"; # Replace with the desired commit.
  py2hwsw_sha256 = "0NstYUgtj02nzFL10XMCooHgfR778XZJVFv8AnaUAEY="; # Replace with the actual SHA256 hash.
  # Get local py2hwsw root from `PY2HWSW_ROOT` env variable
  py2hwswRoot = builtins.getEnv "PY2HWSW_ROOT";

  # For debug
  force_py2_build = 0;

  py2hwsw = 
    # If no root is provided, or there is a root but we want to force a rebuild
    if py2hwswRoot == "" || force_py2_build != 0 then
      pkgs.python3.pkgs.buildPythonPackage rec {
        pname = "py2hwsw";
        version = py2hwsw_commit;
        src =
          if py2hwswRoot != "" then
            # Root provided, use local
            pkgs.lib.cleanSource py2hwswRoot
          else
            # No root provided, use GitHub
            (pkgs.fetchFromGitHub {
              owner = "IObundle";
              repo = "py2hwsw";
              rev = py2hwsw_commit;
              sha256 = py2hwsw_sha256;
              fetchSubmodules = true;
            }).overrideAttrs (_: {
              GIT_CONFIG_COUNT = 1;
              GIT_CONFIG_KEY_0 = "url.https://github.com/.insteadOf";
              GIT_CONFIG_VALUE_0 = "git@github.com:";
            });
        # Add any necessary dependencies here.
        # propagatedBuildInputs = [ pkgs.python38Packages.someDependency ];
      }
    else
      null;

  extra_pkgs = with pkgs; [
    # Define other Nix packages for your project here
    (callPackage ./submodules/VERSAT/versat.nix {})
  ];

in

# If no root is provided, or there is a root but we want to force a rebuild
if py2hwswRoot == "" || force_py2_build != 0 then
  # Use newly built nix package
  import "${py2hwsw}/lib/python${builtins.substring 0 4 pkgs.python3.version}/site-packages/py2hwsw/lib/default.nix" { py2hwsw_pkg = py2hwsw; extra_pkgs = extra_pkgs; }
else
  # Use local
  import "${py2hwswRoot}/py2hwsw/lib/default.nix" { py2hwsw_pkg = py2hwsw; extra_pkgs = extra_pkgs; }
