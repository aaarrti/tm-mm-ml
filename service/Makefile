.PHONY: stubs proto

clean:
	@rm -f -r app/proto/*.py*;
	rm -f -r app/stubs;

all: clean stubs



stubs:
	@openapi-generator generate \
	-i openapi/api.yml \
	-g python-flask \
	--additional-properties=\
	pythonSrcRoot=app,\
	packageName=stubs

