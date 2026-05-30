.PHONY: demo full install-r

demo:
	bash run_all.sh demo

full:
	bash run_all.sh full config/config.yaml config/scRNA_config.yaml

install-r:
	Rscript R/scripts/00_install_packages.R
