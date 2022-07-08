from shared import time_stamp_tag, log_after


@log_after
def name_model(ds) -> str:
    postfix = time_stamp_tag()
    name = f'{ds.name}_la_{postfix}'
    print(f'Named model {name}')
    return name
