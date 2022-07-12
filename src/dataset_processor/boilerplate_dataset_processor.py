import csv
import os
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup, Tag
from typing import List

import collections
import re

from tqdm.auto import tqdm

CASM_MAIN = "CASM_main_content_label"


class BoilerplateDatasetProcessor(ABC):

    @abstractmethod
    def process(self, html_string: str):
        pass

    def process_dataset(self, input_directory_path: str, output_directory_path: str, filename_file_paths: list = None):
        if filename_file_paths:
            filenames = []
            for fp in filename_file_paths:
                with open(fp, "r") as f:
                    filenames += f.readlines()
            filenames = [fname.strip() for fname in filenames]
        else:
            filenames = os.listdir(input_directory_path)

        for fname in tqdm(filenames):
            with open(os.path.join(input_directory_path, fname), "r") as f:
                html_string = f.read()
            processed_html_string = self.process(html_string)

            with open(os.path.join(output_directory_path, fname), "w") as f:
                f.write(processed_html_string)

    def convert_dataset_to_csv(self, html_directory_path: str, csv_output_file_path: str):
        fields = ["filename", "html"]
        html_files = []
        for fname in tqdm(os.listdir(html_directory_path)):
            with open(os.path.join(html_directory_path, fname), 'r') as f:
                source_code = f.read()
            html_files.append([fname, source_code])
        with open(csv_output_file_path, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(html_files)


class AbstractLowestCommonAncestorFinder(ABC):

    def lowest_common_ancestor(self, parents=None, num_of_nodes=0, *args):
        if parents is None:
            parents = collections.defaultdict(int)
        for tag in args:
            parents[tag] += 1
            if parents[tag] == num_of_nodes:
                return tag
        next_arg_list = [tag.parent for tag in args if tag.parent is not None]

        return self.lowest_common_ancestor(parents, num_of_nodes, *next_arg_list)


class TecoDatasetProcessor(BoilerplateDatasetProcessor):
    TECO_MAIN_CONTENT = "TECO_mainContent"
    TECO_NOT_TEMPLATE = "TECO_notTemplate"
    TECO_MAIN_MENU = "TECO_mainMenu"

    def process(self, html_string: str) -> str:
        doc = BeautifulSoup(html_string, "html.parser")
        main_content = doc.find_all(class_=self.TECO_MAIN_CONTENT)

        # mark main content
        for content in main_content:
            content[CASM_MAIN] = 1
            for child in content.descendants:
                if isinstance(child, Tag):
                    child[CASM_MAIN] = 1

        # remove TECO specific markings
        self._remove_class_name(doc, self.TECO_MAIN_CONTENT)
        self._remove_class_name(doc, self.TECO_NOT_TEMPLATE)
        self._remove_class_name(doc, self.TECO_MAIN_MENU)

        return str(doc)

    def _remove_class_name(self, doc, class_name):
        class_name = class_name.lower()
        nodes = doc.find_all(class_=class_name)
        for node in nodes:
            node["class"].remove(class_name)


# For cleaneval and googletrends-2017
class BoilernetDatasetProcessor(BoilerplateDatasetProcessor):
    BOILERNET_LABEL = "__boilernet_label"

    def process(self, html_string: str) -> str:
        doc = BeautifulSoup(html_string, "html.parser")
        main_content = doc.find_all(attrs={self.BOILERNET_LABEL: 1})

        # mark main content
        for content in main_content:
            content[CASM_MAIN] = 1
            for child in content.descendants:
                if isinstance(child, Tag):
                    child[CASM_MAIN] = 1

        # unwrap Boilernet added span
        self._unwrap_boilernet_span(doc, self.BOILERNET_LABEL)
        # remove Boilernet specific markings
        self._remove_attribute(doc, self.BOILERNET_LABEL)

        return str(doc)

    def _unwrap_boilernet_span(self, doc: BeautifulSoup, attribute_name: str):
        attribute_name = attribute_name.lower()
        nodes = doc.find_all(attrs={attribute_name: 1})
        nodes += doc.find_all(attrs={attribute_name: 0})
        for node in nodes:
            if type(node) == Tag and node.name == "span":
                if node.string:
                    node.string.replace_with(node.string.strip())
                if node.parent and node.get(CASM_MAIN):
                    node.parent[CASM_MAIN] = node.get(CASM_MAIN)
                node.unwrap()

    def _remove_attribute(self, doc, attribute_name):
        attribute_name = attribute_name.lower()
        nodes = doc.find_all(attrs={attribute_name: 1})
        nodes += doc.find_all(attrs={attribute_name: 0})
        for node in nodes:
            del node[attribute_name]


# For cleaneval and googletrends-2017
class BlockLevelBoilernetDatasetProcessor(BoilernetDatasetProcessor, AbstractLowestCommonAncestorFinder):

    def process(self, html_string: str, html_id: str = None) -> str:
        doc = BeautifulSoup(html_string, "html.parser")
        main_content = doc.find_all(attrs={self.BOILERNET_LABEL: 1})

        try:
            lca = self.lowest_common_ancestor(None, len(main_content), *main_content)
        except RecursionError as e:
            to_print = html_id if html_id else html_string
            print(f"Recursion error for {to_print}: {e}")
            return str(doc)

        # mark main content block and all its descendants
        lca[CASM_MAIN] = 1
        has_label_0 = False
        for child in lca.descendants:
            if isinstance(child, Tag):
                child[CASM_MAIN] = 1
                if not has_label_0 and child.get(self.BOILERNET_LABEL) == "0":
                    to_print = html_id if html_id else html_string
                    print(f"Warning for {to_print}: Boilernet label 0 exists inside main content block.")
                    has_label_0 = True

        # unwrap Boilernet added span
        self._unwrap_boilernet_span(doc, self.BOILERNET_LABEL)
        # remove Boilernet specific markings, if there's still any
        self._remove_attribute(doc, self.BOILERNET_LABEL)

        return str(doc)


class DragnetDatasetProcessor(BoilerplateDatasetProcessor):
    COMMENTS = "!@#$%^&*()  COMMENTS"

    def __init__(self, num_chars_to_find: int = 50):
        self.num_chars_to_find = num_chars_to_find

    def process(self, html_file_path: str, main_content_file_path: str, num_chars_to_find: int = 50,
                include_comments: bool = False):
        num_chars_to_find = self.num_chars_to_find if self.num_chars_to_find else num_chars_to_find
        html_string, main_content_strings = self._open_files(html_file_path, main_content_file_path, include_comments)
        doc = BeautifulSoup(html_string, "html.parser")
        main_content = self.get_main_content_elements(doc, main_content_strings, num_chars_to_find=num_chars_to_find)

        # mark main content
        for content in main_content:
            content[CASM_MAIN] = 1
            for child in content.descendants:
                if isinstance(child, Tag):
                    child[CASM_MAIN] = 1

        return str(doc)

    def _open_files(self, html_file_path: str, main_content_file_path: str, include_comments: bool = False):
        with open(html_file_path, "rb") as f:
            html_string = f.read()
        with open(main_content_file_path, "r") as f:
            try:
                main_content_strings = f.readlines()
            except UnicodeDecodeError:
                with open(main_content_file_path, "r", encoding="cp1252") as fi:
                    main_content_strings = fi.readlines()
            main_content_strings = [c.strip() for c in main_content_strings if c.strip() != ""]
            if not include_comments:
                try:
                    comments_idx = main_content_strings.index(self.COMMENTS)
                    main_content_strings = main_content_strings[:comments_idx]
                except ValueError:
                    pass
        return html_string, main_content_strings

    def get_main_content_elements(self, html_doc: BeautifulSoup, main_contents: List[str],
                                  num_chars_to_find: int = 50) -> List[Tag]:
        elements = []
        for content in main_contents:
            text_to_search_for = content[:num_chars_to_find].replace('"', '')
            els = html_doc.select(':-soup-contains-own("' + text_to_search_for + '")')
            els += html_doc.find_all("", string=re.compile(re.escape(content[:num_chars_to_find])))
            for el in els:
                if ' '.join(str(el).split()) != ' '.join(content.split()):
                    els.remove(el)
            elements.extend([el.parent for el in els if el.parent not in elements])
        return elements


class BlockLevelDragnetDatasetProcessor(DragnetDatasetProcessor, AbstractLowestCommonAncestorFinder):
    def process(self, html_file_path: str = None, main_content_file_path: str = None,
                html_string: str = None, main_content_strings: List[str] = None,
                num_chars_to_find: int = 50,
                include_comments: bool = False, html_id: str = None):
        num_chars_to_find = self.num_chars_to_find if self.num_chars_to_find else num_chars_to_find
        if html_string is None or main_content_strings is None:
            html_string, main_content_strings = self._open_files(html_file_path, main_content_file_path,
                                                                 include_comments)
        doc = BeautifulSoup(html_string, "html.parser")
        main_content = self.get_main_content_elements(doc, main_content_strings, num_chars_to_find=num_chars_to_find)

        # FIXME
        # common_parents = self.common_parents(main_content)

        try:
            lca = self.lowest_common_ancestor(None, len(main_content), *main_content)
        except RecursionError as e:
            to_print = html_id if html_id else html_string
            print(f"Recursion error for {to_print}: {e}")
            return str(doc)

        # mark main content block and all its descendants
        lca[CASM_MAIN] = 1
        for child in lca.descendants:
            if isinstance(child, Tag):
                child[CASM_MAIN] = 1

        return str(doc)

    def lowest_common_ancestor(self, parents=None, num_of_nodes=0, *args):
        if parents is None:
            parents = collections.defaultdict(int)
        for tag in args:
            parents[tag] += 1
            if parents[tag] == num_of_nodes:
                return tag
        next_arg_list = [tag.parent for tag in args if tag.parent is not None]

        return self.lowest_common_ancestor(parents, num_of_nodes, *next_arg_list)

    # FIXME
    def common_parents(self, main_content_elements: List[Tag]):
        main_content_strings = [string for element in main_content_elements for string in element.strings]
        i = 0
        while i < len(main_content_elements):
            first = main_content_elements[i]
            second = main_content_elements[i + 1]
            # check if second is descendant of first
            if second in first.descendants:
                del main_content_elements[i + 1]
            else:
                # find common parents between first and second
                common_parent = self.lowest_common_ancestor(None, 2, first, second)

                has_boilerplate = False
                for string in common_parent.strings:
                    if string not in main_content_strings:
                        has_boilerplate = True
                        break

                if not has_boilerplate:
                    main_content_elements[i] = common_parent
                    del main_content_elements[i + 1]
            i += 1


# class LabelledDatasetProcessor:
#
#     def __init__(self, tokenizer: RobertaTokenizer, label_attribute_name: str = CASM_MAIN):
#         self.tokenizer = tokenizer
#         self.label_attribute_name = label_attribute_name.lower()
#
#     def process(self, html_string: str, **kwargs):
#         doc_with_label = BeautifulSoup(html_string, "html.parser")
#         all_elements = doc_with_label.find_all()
#
#         doc_without_label = doc_with_label.__copy__()
#         self._remove_attribute(doc_without_label, self.label_attribute_name)
#
#         tokenized_doc = self.tokenizer(str(doc_without_label), **kwargs)
#         labels = []
#         i = 0
#         prev_node_id = 1  # node index starts at 2
#         for node_id in tokenized_doc["node_indices"]:
#             if prev_node_id != 1 and prev_node_id != node_id:
#                 i += 1
#             if node_id == self.tokenizer.pad_token_id:
#                 labels.append(-100)
#             elif i < len(all_elements) and self._element_has_positive_label(all_elements[i]):
#                 labels.append(1)
#             else:
#                 labels.append(0)
#             prev_node_id = node_id
#
#         tokenized_doc["labels"] = labels
#         return tokenized_doc
#
#     def _element_has_positive_label(self, element: Tag):
#         if element.has_attr(self.label_attribute_name) and int(element[self.label_attribute_name]) == 1:
#             return True
#         return False
#
#     def _remove_attribute(self, doc, attribute_name):
#         attribute_name = attribute_name.lower()
#         nodes = doc.find_all(attrs={attribute_name: 1})
#         nodes += doc.find_all(attrs={attribute_name: 0})
#         for node in nodes:
#             del node[attribute_name]