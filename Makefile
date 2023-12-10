@all:
	
local:
	jupyter notebook

public:
	jupyter notebook --ip=0.0.0.0 --port=8080
