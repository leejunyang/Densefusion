# 核心配置
PYTHON = python
PIP = pip
BUILD_DIR := build
KNN_LIB := torch_knn*.so

.PHONY: all install develop test clean

all: install  ## 默认执行安装

install:  ## 安装到当前Conda环境（需提前激活）
	@echo "Detecting current env: $(shell which python)"
	@$(PYTHON) -c "import torch; print(f'PyTorch version: {torch.__version__}')" 
	$(PIP) install .
	@echo "\nValidating installation: make test"

develop:  ## 开发者模式安装（可编辑模式）
	$(PYTHON) setup.py develop

test:  ## 运行安装验证
	@$(PYTHON) -c "import torch_knn, torch; print('CUDA可用:', torch.cuda.is_available())"

clean:  ## 清除编译产物
	rm -rf $(BUILD_DIR) $(KNN_LIB)
	find . -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist
	@echo "Uninstall packages please execuate: make uninstall"

uninstall:  ## 从当前环境卸载包
	$(PIP) uninstall torch_knn -y