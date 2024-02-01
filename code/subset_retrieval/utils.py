import json

def read_CSN_examples(filename, data_num=-1):
    """Read examples from CSN filename. data_num=-1 for all examples."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if data_num != -1 and idx + 1 == data_num:
                break
    return examples

def read_project_examples(filename):
    """Read examples from project-code2nl filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)['translation']
            code = js['code']
            nl = js['nl']
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )

    return examples

class Example(object):
    """A single training/test example."""

    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target



def convert_examples_to_src_ids(item):
    example, tokenizer, args = item

    source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1

    return source_ids

def convert_examples_to_tgt_ids(item):
    example, tokenizer, args = item

    target_str = example.target

    target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length', truncation=True)

    return target_ids
