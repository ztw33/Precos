import sys
import lucene
import re
from java.nio.file import Paths
from java.lang import Integer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, StoredField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig,IndexOptions,DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanQuery
from org.apache.lucene.queryparser.classic import QueryParser
from tqdm import tqdm
import argparse
from utils import *
import os

lucene.initVM()


def build_index(datalist, output_dir):
    os.system(f"mkdir -p {output_dir}")
    indexDir = SimpleFSDirectory(Paths.get(output_dir+"/lucene_index/"))
    config = IndexWriterConfig(WhitespaceAnalyzer())
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(indexDir, config)

    if writer.numDocs():
        print("Index already built.")
        return
    
    for id, data in enumerate(tqdm(datalist, desc="Building index")):
        doc = Document()
        doc.add(StoredField("id", str(id)))
        doc.add(TextField("data", data, Field.Store.YES))

        writer.addDocument(doc)
    
    print("Closing index of %d docs..." % writer.numDocs())
    writer.close()
    

def retriever(query_list, k, index_dir, output_dir):
    os.system(f"mkdir -p {output_dir}")
    analyzer = WhitespaceAnalyzer()
    reader = DirectoryReader.open(SimpleFSDirectory(Paths.get(index_dir+"/lucene_index/")))
    searcher = IndexSearcher(reader)
    queryParser = QueryParser("data", analyzer)
    BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE)

    output_filepath = f"{output_dir}/subset_idx.txt"
    with open(output_filepath, 'w+') as fout:
        for i, query in enumerate(tqdm(query_list, desc="Querying")):
            qp = queryParser.parse(QueryParser.escape(query))
            hits = searcher.search(qp, k).scoreDocs

            if len(hits) == 0:
                print("No results found for query %d" % i)
                fout.write('\n')
            
            else:
                fout.write(' '.join([str(eval(searcher.doc(hit.doc).get("id"))) for hit in hits]) + '\n')
            
    reader.close()


def add_args(parser):
    parser.add_argument("--build_index", action='store_true',
                        help="Building lucene index.")
    parser.add_argument("--subset", action='store_true',
                        help="Retrieving subset.")
    
    parser.add_argument("--key", default="code", type=str,
                        help="code or nl.")
    
    parser.add_argument("--data_filename", default=None, type=str,
                        help="Data path for building index.")
    
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the results will be written.")
    parser.add_argument("--index_dir", default=None, type=str,
                        help="The directory where the index is stored.")

    parser.add_argument("--topk", default=3, type=int,
                        help="Top-k neighbors when retrieving.")
    
    return parser.parse_args()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    if args.build_index:
        examples = read_project_examples(args.data_filename)
        datalist = [e.target if args.key == "nl" else e.source for e in examples]
        build_index(datalist, args.output_dir)

    elif args.subset:
        examples = read_project_examples(args.data_filename)
        datalist = [e.target if args.key == "nl" else e.source for e in examples]
        retriever(datalist, args.topk, args.index_dir, args.output_dir)
