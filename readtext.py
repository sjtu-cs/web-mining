 #! /usr/bin/env python
 positive_examples = list(open('/data/train.txt', "r").readlines())
 positive_examples = [s.strip() for s in positive_examples]
 labels = [s[0] for s in positive_examples]
 records = [s[2:] for s in positive_examples]
 print(records)