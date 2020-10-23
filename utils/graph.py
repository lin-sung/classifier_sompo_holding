#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: graph.py
ultilities for graph
"""


__author__ = "Ethan and Marc"


import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = 'NotoSansCJK-Black.ttc'

def check_intersect_range(coord1, length1, coord2, length2):
    if coord1 > coord2:
        coord1, coord2 = coord2, coord1
        length1, length2 = length2, length1

    return (coord1 + length1) > coord2


def get_intersect_range(coord1, length1, coord2, length2):
    if coord1 > coord2:
        coord1, coord2 = coord2, coord1
        length1, length2 = length2, length1

    if not check_intersect_range(coord1, length1, coord2, length2):
        return 0

    if (coord1 + length1) > (coord2 + length2):
        return length2
    else:
        return coord1 + length1 - coord2


def check_intersect_horizontal_proj(cell1, cell2):
    return check_intersect_range(cell1.y, cell1.h, cell2.y, cell2.h)


def check_intersect_vertical_proj(cell1, cell2):
    return check_intersect_range(cell1.x, cell1.w, cell2.x, cell2.w)


def get_intersect_range_horizontal_proj(cell1, cell2):
    return get_intersect_range(cell1.y, cell1.h, cell2.y, cell2.h)


def get_intersect_range_vertical_proj(cell1, cell2):
    return get_intersect_range(cell1.x, cell1.w, cell2.x, cell2.w)


class CellNode(object):

    threshhold_really_horizontal = 2.0
    threshold_really_vertical = 0.2

    def __init__(self, ref_image, x, y, w, h, label=None, cell_type=None, is_sub=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xc = self.x + w // 2
        self.yc = self.y + h // 2
        self.is_sub = is_sub
        self.parent = None
        self.drawed_rel = False
        self.label = label
        self.cell_type = cell_type

        # Get cell background

        self.list_texts = []
        self.sub_lines = []

        # Init adjacent
        self.lefts = []
        self.rights = []
        self.tops = []
        self.bottoms = []
        self.is_master_key = False
        self.is_value = False
        self.parent_key = []
        self.children_key = []
        self.drawed = False
        self.text = ""
        self.is_column_header = False
        self.column_header = None
        self.col = None
        self.row = None
        self.confidence = None

    def __getitem__(self, idx):
        return (self.x, self.y, self.w, self.h)[idx]

    def get_text(self):
        return self.text
        # if self.is_sub:
        #     return "".join([text for text in self.list_texts])
        # else:
        #     return " ".join([subline.get_text() for subline in self.sub_lines])

    def get_aspect_ratio(self):
        return self.w / self.h

    def is_really_horizontal_cell(self):
        return self.get_aspect_ratio() > CellNode.threshhold_really_horizontal

    def is_really_vertical_cell(self):
        return self.get_aspect_ratio() < CellNode.threshold_really_vertical

    def get_bbox(self):
        return [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h),
        ]

    def get_real_sub_lines(self):
        if len(self.sub_lines) == 0:
            if self.is_sub:
                return [self]
            else:
                return []
        else:
            return_list = []

            if self.is_sub:
                return_list.append(self)

            for child in self.sub_lines:
                return_list.extend(child.get_real_sub_lines())

            return return_list

    def is_left_of(self, other_cell, ref_cells):
        """Check if this cell is directly left of
        other cell given a set of full cells sorted"""
        # 0. Check if other_cell in self.rights

        if other_cell in self.rights:
            return True

        if other_cell.x < self.x or not check_intersect_horizontal_proj(
            self, other_cell
        ):
            return False

        if get_intersect_range_horizontal_proj(self, other_cell) > 0.9 * min(
            self.h, other_cell.h
        ):
            if other_cell.x - self.x < 0.1 * min(self.w, other_cell.w):
                return True

        # => right now: other cell is on the right and intersect on projection
        # horizontal side

        if len(ref_cells) == 0:
            return True

        # 1. get all cells on the right of this cell.
        # meaning all cells that have overlapping regions with this cell
        # and lie to the right
        ref_cells = [
            cell

            for cell in ref_cells

            if check_intersect_horizontal_proj(self, cell)
            and (cell.x + cell.w) < other_cell.x + other_cell.w * 0.1
            and cell.x >= (self.x + self.w * 0.8)
            and check_intersect_horizontal_proj(self, cell)
        ]
        # 2. filters all the small overlapping cells
        ref_cells = [
            cell

            for cell in ref_cells

            if get_intersect_range_horizontal_proj(self, cell) > min(self.h, cell.h) / 5
        ]
        ref_cells = [
            cell

            for cell in ref_cells

            if get_intersect_range_horizontal_proj(cell, other_cell) > other_cell.h / 2
            or get_intersect_range_horizontal_proj(self, cell)
            > min(cell.h, self.h) * 0.8
        ]

        # 3. Check if there are any cells lies between this and other_cell

        if len(ref_cells) > 0:
            return False

        # 4. return results

        return True

    def is_right_of(self, other_cell, ref_cells):
        return other_cell.is_left_of(self, ref_cells)

    def is_top_of(self, other_cell, ref_cells):
        """Check if this cell is directly top of
        other cell given a set of full cells sorted"""
        # 0. Check if other_cell in self.rights

        if other_cell in self.bottoms:
            return True

        if other_cell.y < self.y or not check_intersect_vertical_proj(self, other_cell):
            return False

        if (
            get_intersect_range_vertical_proj(self, other_cell)
            < min(self.w, other_cell.w) / 5
        ):
            return False
        # => right now: other cell is on the right and intersect on projection
        # horizontal side

        if len(ref_cells) == 0:
            return True

        # 1. get all cells on the right of this cell.
        # meaning all cells that have overlapping regions with this cell
        # and lie to the right
        ref_cells = [
            cell

            for cell in ref_cells

            if check_intersect_vertical_proj(self, cell)
            and (cell.y + cell.h) < other_cell.y + other_cell.h * 0.1
            and cell.y >= (self.y + self.h * 0.8)
            and check_intersect_vertical_proj(self, cell)
        ]
        # 2. filters all the small overlapping cells
        ref_cells = [
            cell

            for cell in ref_cells

            if get_intersect_range_vertical_proj(self, cell) > min(self.w, cell.w) / 5
        ]
        ref_cells = [
            cell

            for cell in ref_cells

            if get_intersect_range_vertical_proj(cell, other_cell) > other_cell.w / 2
            or get_intersect_range_vertical_proj(self, cell) > min(self.w, cell.w) * 0.8
        ]

        # 3. Check if there are any cells lies between this and other_cell

        if len(ref_cells) > 0:
            return False

        # 4. return result

        return True

    def set_text(self, text):
        self.ocr_value = text
        self.list_texts = [text]


def get_cell_from_cell_list(cell_list, img=None):
    out_cell_list = []
    for idx, cell in enumerate(cell_list):
        x1, y1, x2, y2 = cell[:4]
        category = int(cell[4])
        text = cell[6]
        text_BoW = cell[7]
        type = cell[5]

        new_cell = CellNode(
            img,
            x1,
            y1,
            x2 - x1 + 1,
            y2 - y1 + 1,
            category,
            type,
        )
        new_cell.name = idx
        new_cell.text = text
        new_cell.text_BoW = text_BoW
        new_cell.list_texts = [text]

        out_cell_list.append(new_cell)

    return out_cell_list


def get_cell_from_cell_dict(cell_dict, img):
    table_cell_name_re = re.compile("table[0-9]+_cell[0-9]+$")
    cell_list = []
    list_sub_cell = []

    for key in cell_dict:
        # Initialize new cell

        if key == "form_type":
            continue

        cell_bbox = cell_dict[key]["location"]

        if type(cell_bbox[0]) is list:
            xs = [p[0] for p in cell_bbox]
            ys = [p[1] for p in cell_bbox]
            cur_x = min(xs)
            cur_y = min(ys)
            cell_bbox = [cur_x, cur_y, max(xs) - cur_x, max(ys) - cur_y]

        cell_label = cell_dict[key].get("label")
        cell_type = cell_dict[key].get("type")
        #for debug purpose
        # cell_bbox = [cell_bbox[0][0], cell_bbox[0][1], cell_bbox[2][0] - cell_bbox[0][0], cell_bbox[2][1] - cell_bbox[0][1]]
        new_cell = CellNode(
            img,
            cell_bbox[0],
            cell_bbox[1],
            cell_bbox[2] + 1,
            cell_bbox[3] + 1,
            cell_label,
            cell_type,
        )
        new_cell.ocr_value = cell_dict[key].get("value", "")
        new_cell.list_texts = [cell_dict[key].get("value", "")]
        new_cell.confidence = cell_dict[key].get("confidence", "")
        
        if new_cell.w < 10 or new_cell.h < 10:
            continue

        if table_cell_name_re.match(key) is None:
            list_sub_cell.append(new_cell)
            list_sub_cell[-1].name = key
            list_sub_cell[-1].is_sub = True
        else:
            cell_list.append(new_cell)
            cell_list[-1].name = key
    i = 0
    final_lines_with_no_parents = []

    while len(list_sub_cell) > 0:
        is_collied = False

        for cell in cell_list:
            if len(list_sub_cell) == 0:
                break

            if (
                cell.name.split("_")[0] == list_sub_cell[i].name.split("_")[0]
                and cell.name.split("_")[1] == list_sub_cell[i].name.split("_")[1]
            ):
                cell.sub_lines.append(list_sub_cell.pop(i))
                cell.sub_lines[-1].parent = cell
                is_collied = True

        if len(list_sub_cell) == 0:
            break

        if not is_collied:
            final_lines_with_no_parents.append(list_sub_cell.pop(i))
    cell_list.extend(final_lines_with_no_parents)

    return cell_list


def _get_v_intersec(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    y1_2 = y1_1 + h1
    y2_2 = y2_1 + h2
    ret = max(0, min(y1_2 - y2_1, y2_2 - y1_1))

    return ret


def _get_v_union(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    y1_2 = y1_1 + h1
    y2_2 = y2_1 + h2
    ret = min(h1 + h2, max(y2_2 - y1_1, y1_2 - y2_1))

    return ret


def _get_h_intersec(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    x1_2 = x1_1 + w1
    x2_2 = x2_1 + w2
    ret = max(0, min(x1_2 - x2_1, x2_2 - x1_1))

    return ret


def _get_h_union(loc1, loc2):
    x1_1, y1_1, w1, h1 = loc1
    x2_1, y2_1, w2, h2 = loc2
    x1_2 = x1_1 + w1
    x2_2 = x2_1 + w2
    ret = min(w1 + w2, max(x2_2 - x1_1, x1_2 - x2_1))

    return ret


def get_nearest_line(cr_line, list_lines, dr="l", thresh=50000):
    ret = None
    dt = thresh

    for line in list_lines:
        text = line.get_text()

        if text == "":
            continue

        if dr == "r" or dr == "l":
            if _get_v_intersec(line, cr_line) <= 0.3 * _get_v_union(line, cr_line):
                continue
        elif dr == "t" or dr == "b":
            d = min(
                abs(line.y - cr_line.y - cr_line.h), abs(cr_line.y - line.y - line.h)
            )

            # 0.1 * _get_h_union(loc, cr_line):

            if _get_h_intersec(line, cr_line) <= 0:
                if dr == "t" and cr_line.y > line.y:
                    continue

                if not (
                    dr == "t"
                    and d < 0.5 * cr_line.h
                    and cr_line.y + 1.3 * cr_line.h > line.y
                ):
                    continue

        dist = dt + 1

        if dr == "r":
            if line.x > cr_line.x:
                dist = line.x - cr_line.x - cr_line.w
        elif dr == "l":
            if line.x < cr_line.x:
                dist = cr_line.x - line.x - line.w
        elif dr == "b":
            if line.y > cr_line.y:
                dist = line.y - cr_line.y - cr_line.h
        elif dr == "t":
            if line.y < cr_line.y:
                dist = cr_line.y - line.y - line.h

        if dist < dt:
            ret = line
            dt = dist

    return ret


class Edge:
    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label


class Column(object):
    def __init__(self, cell_list):
        self.cell_list = cell_list
        self.x = min([c.x for c in cell_list])
        self.y = min([c.y for c in cell_list])
        self.w = max([c.x + c.w - self.x for c in cell_list]) #cell_list[0].w
        self.h = sum([c.h for c in cell_list])


class Row(object):
    def __init__(self, cell_list):
        self.cell_list = cell_list
        self.x = min([c.x for c in cell_list])
        self.y = min([c.y for c in cell_list])
        self.h = max([c.y + c.h - self.y for c in cell_list]) #cell_list[0].h
        self.w = sum([c.w for c in cell_list])


class Graph():

    edge_labels = ["lr", "rl", "tb", "bt", "child", "parent"]

    def __init__(self, cell_list, doc_img = None):
        self.doc_img = doc_img
        self.org_items = get_cell_from_cell_list(cell_list, doc_img)

        self.table_cells = [item for item in self.org_items if not item.is_sub]
        self.text_lines = []

        # for item in self.org_items:
        #     self.text_lines.extend(item.get_real_sub_lines())
        # self._detect_row()
        # self._detect_column()

        self.rows = self.cols = []

        self.nodes = self.table_cells + self.text_lines + self.rows + self.cols
        # self.nodes = self.table_cells + self.text_lines
        """
        print([c.name for c in self.table_cells])
        print([c.name for c in self.text_lines])
        print('---------------------------------------')
        """
        self.edges = []
        self.build_edges()
        self._get_adj_matrix()

    # def __getstate__(self):
    #     #print('Loading pickle state')
    #     return self.doc_img, self.org_items, self.table_cells, self.text_lines, self.nodes, self.edges, self.cols, self.rows, self.adj

    # def __setstate__(self, state):
    #     #print('Saving pickle state')
    #     self.doc_img, self.org_items, self.table_cells, self.text_lines, self.nodes, self.edges, self.cols, self.rows, self.adj = state

    def build_edges(self):
        for cell_list in (self.text_lines, self.table_cells):
            # 1. left-right
            cell_list_top_down = sorted(cell_list, key=lambda cell: cell.y)
            cell_list_left_right = sorted(cell_list, key=lambda cell: cell.x)
            # 1.1 Check this cell with every cell to the right of it
            # TODO: More effective iteration algo e.g: cached collisions matrix
            self._build_left_right_edges_1(cell_list_top_down)
            # 2. top-down
            self._build_top_down_edges_1(cell_list_left_right)
        self._build_child_parent_edges()

        # clean left-right edges
        self._clean_left_right_edges()
        # clean top-bot edges
        # self._clean_top_bot_edges()

    def _build_left_right_edges(self, cell_list_top_down):
        for cell in cell_list_top_down:
            cell_collide = [
                other_cell

                for other_cell in cell_list_top_down

                if other_cell.x >= cell.x
                and check_intersect_horizontal_proj(cell, other_cell)
                and cell != other_cell
            ]
            cell_collide = [
                other_cell

                for other_cell in cell_collide

                if get_intersect_range_horizontal_proj(cell, other_cell)
                > min(cell.h, other_cell.h) * 0.4
            ]

            for other_cell in cell_collide:
                if (
                    cell.is_left_of(other_cell, cell_collide)
                    and other_cell not in cell.rights
                ):
                    self.edges.append(
                        Edge(cell, other_cell, self.edge_labels.index("lr"))
                    )
                    self.edges.append(
                        Edge(other_cell, cell, self.edge_labels.index("rl"))
                    )
                    cell.rights.append(other_cell)
                    other_cell.lefts.append(cell)

        """
        def get_rights(cell):
            ret = [cell]
            if len(cell.rights) == 0:
                return ret
            for r in cell.rights:
                ret.extend(get_rights(r))
            return list(set(ret))
        # extend the connection
        for cell in cell_list_top_down:
            if len(cell.lefts) > 0:
                continue
            # get list of right cells
            rights = get_rights(cell)
            for r in rights:
                if r in cell.rights or r == cell:
                    continue
                if not get_intersect_range_horizontal_proj(cell, r) > min(cell.h, r.h) * 0.3:
                    continue
                self.edges.append(
                    Edge(cell, r, self.edge_labels.index('lr')))
                self.edges.append(
                    Edge(r, cell, self.edge_labels.index('rl')))
                cell.rights.append(r)
                r.lefts.append(cell)
        """

    def _clean_left_right_edges(self):
        for cell in self.text_lines:
            if len(cell.lefts) <= 1:
                continue
            left_cells = sorted(cell.lefts, key=lambda x: x.x)
            removes = [
                c

                for c in left_cells

                if c.x + c.w > cell.x and c.x > cell.x - 0.5 * cell.h
            ]
            left_cells = [c for c in left_cells if c not in removes]
            # cluster these cell into column:

            columns = []
            column_cells = []
            # column_x = left_cells[0].x

            for c in left_cells:
                its = 0
                union = 100

                if column_cells:
                    its = get_intersect_range_vertical_proj(column_cells[-1], c)
                    union = min(column_cells[-1].w, c.w)

                if its > 0.5 * union:
                    column_cells.append(c)

                    continue
                else:
                    if len(column_cells) > 0:
                        columns.append(column_cells)
                    column_cells = [c]
                    # column_x = c.x

            if column_cells:
                columns.append(column_cells)

            # left_cells to keep:

            if len(columns) > 0:
                real_lefts = columns[-1]
            else:
                real_lefts = []
            removes += [c for c in left_cells if c not in real_lefts]
            remove_edges = []

            for c in removes:
                c.rights.remove(cell)

                for e in self.edges:
                    if (
                        e.start == c
                        and e.end == cell
                        and e.label == self.edge_labels.index("lr")
                    ):
                        remove_edges.append(e)

                    if (
                        e.start == cell
                        and e.end == c
                        and e.label == self.edge_labels.index("rl")
                    ):
                        remove_edges.append(e)
            [self.edges.remove(e) for e in remove_edges]

            cell.lefts = real_lefts

    def _build_top_down_edges(self, cell_list_left_right):
        for cell in cell_list_left_right:
            cell_collide = [
                other_cell

                for other_cell in cell_list_left_right

                if other_cell.y > cell.y + cell.h * 0.6
                and check_intersect_vertical_proj(cell, other_cell)
                and cell != other_cell
            ]

            for other_cell in cell_collide:
                if (
                    cell.is_top_of(other_cell, cell_collide)
                    and other_cell not in cell.bottoms
                ):
                    self.edges.append(
                        Edge(cell, other_cell, self.edge_labels.index("tb"))
                    )
                    self.edges.append(
                        Edge(other_cell, cell, self.edge_labels.index("bt"))
                    )
                    cell.bottoms.append(other_cell)
                    other_cell.tops.append(cell)
        """
        def get_bottoms(cell):
            ret = [cell]
            if len(cell.bottoms) == 0:
                return ret
            print(cell.name, [c.name for c in cell.bottoms])
            for b in cell.bottoms:
                ret.extend(get_bottoms(b))
            return list(set(ret))
        # extend the connection
        for cell in cell_list_left_right:
            # get list of right cells
            bottoms = get_bottoms(cell)
            print(cell, len(bottoms), '--------------------------')
            for b in bottoms:
                if b in cell.bottoms or b == cell:
                    continue
                if not get_intersect_range_vertical_proj(cell, b) > min(cell.w, b.w) * 0.3:
                    continue
                self.edges.append(
                    Edge(cell, b, self.edge_labels.index('tb')))
                self.edges.append(
                    Edge(b, cell, self.edge_labels.index('bt')))
                cell.bottoms.append(b)
                b.tops.append(cell)
        """

    def _build_top_down_edges_1(self, cell_list_left_right):
        for cell in cell_list_left_right:
            top_cell = get_nearest_line(cell, cell_list_left_right, "t")

            if top_cell:
                self.edges.append(Edge(top_cell, cell, self.edge_labels.index("tb")))
                self.edges.append(Edge(cell, top_cell, self.edge_labels.index("bt")))
                cell.tops.append(top_cell)
                if len(top_cell.bottoms) == 0:
                    top_cell.bottoms.append(cell)

    def _build_left_right_edges_1(self, cell_list_top_down):
        for cell in cell_list_top_down:
            left_cell = get_nearest_line(cell, cell_list_top_down, "l")

            if left_cell:
                self.edges.append(Edge(left_cell, cell, self.edge_labels.index("lr")))
                self.edges.append(Edge(cell, left_cell, self.edge_labels.index("rl")))
                cell.lefts.append(left_cell)
                if len(left_cell.rights) == 0:
                    left_cell.rights.append(cell)

    def _clean_top_bot_edges(self):
        for cell in self.text_lines:
            if len(cell.tops) <= 1:
                continue
            top_cells = sorted(cell.tops, key=lambda x: x.y)

            rows = []
            row_cells = []

            for c in top_cells:
                its = 0
                union = 10000

                if row_cells:
                    its = get_intersect_range_horizontal_proj(row_cells[-1], c)
                    union = min(row_cells[-1].w, c.w)

                if its > 0.5 * union:
                    row_cells.append(c)

                    continue
                else:
                    if len(row_cells) > 0:
                        rows.append(row_cells)
                    row_cells = [c]

            if row_cells:
                rows.append(row_cells)

            # left_cells to keep:
            real_tops = rows[-1]
            removes = [c for c in top_cells if c not in real_tops]
            remove_edges = []

            for c in removes:
                c.bottoms.remove(cell)

                for e in self.edges:
                    if (
                        e.start == c
                        and e.end == cell
                        and e.label == self.edge_labels.index("tb")
                    ):
                        remove_edges.append(e)

                    if (
                        e.start == cell
                        and e.end == c
                        and e.label == self.edge_labels.index("bt")
                    ):
                        remove_edges.append(e)
            [self.edges.remove(e) for e in remove_edges]

            cell.tops = real_tops

    def _build_child_parent_edges(self):
        for cell in self.text_lines:
            parent = cell.parent

            if not parent:
                continue
            childs = parent.sub_lines

            for ch in childs:
                self.edges.append(Edge(ch, parent, self.edge_labels.index("parent")))
                self.edges.append(Edge(parent, ch, self.edge_labels.index("child")))

        for row in self.rows:
            for ch in row.cell_list:
                self.edges.append(Edge(ch, row, self.edge_labels.index("parent")))
                self.edges.append(Edge(row, ch, self.edge_labels.index("child")))

        for col in self.cols:
            for ch in col.cell_list:
                self.edges.append(Edge(ch, col, self.edge_labels.index("parent")))
                self.edges.append(Edge(col, ch, self.edge_labels.index("child")))

    def _detect_column(self):
        self.cols = []
        used_cells = []

        for cell in self.table_cells:
            aligns = []

            if cell in used_cells:
                continue
            aligns.append(cell)
            p_margin = cell.w / 4
            w_margin = cell.w / 6

            for o_cell in self.table_cells:
                if o_cell in used_cells:
                    continue

                if o_cell == cell:
                    continue

                if (
                    abs(o_cell.x - cell.x) <= p_margin
                    and abs(o_cell.w - cell.w) <= w_margin
                ):
                    aligns.append(o_cell)
            used_cells.extend(aligns)

            if len(aligns) > 1:
                new_col = Column(aligns)

                for c in aligns:
                    c.col = new_col
                self.cols.append(new_col)

    def _detect_row(self):
        self.rows = []
        used_cells = []

        for cell in self.table_cells:
            aligns = []

            if cell in used_cells:
                continue
            aligns.append(cell)
            p_margin = cell.h / 2
            h_margin = cell.h / 4

            for o_cell in self.table_cells:
                if o_cell in used_cells:
                    continue

                if o_cell == cell:
                    continue

                if (
                    abs(o_cell.y - cell.y) <= p_margin
                    and abs(o_cell.h - cell.h) <= h_margin
                ):
                    aligns.append(o_cell)
            used_cells.extend(aligns)

            if len(aligns) > 1:
                new_row = Row(aligns)

                for c in aligns:
                    c.row = new_row
                self.rows.append(new_row)

    def _get_adj_matrix(self):
        def scale_coor(node):
            scale_x1 = (node.x - min_x) / max_delta_x
            scale_y1 = (node.y - min_y) / max_delta_y
            scale_x1b = (node.x + node.w - min_x) / max_delta_x
            scale_y1b = (node.y + node.h - min_y) / max_delta_y

            return scale_x1, scale_y1, scale_x1b, scale_y1b

        def dist(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        def rect_distance(rect1, rect2):
            x1, y1, x1b, y1b = rect1
            x2, y2, x2b, y2b = rect2

            left = x2b < x1
            right = x1b < x2
            bottom = y2b < y1
            top = y1b < y2

            if top and left:
                return dist(x1, y1b, x2b, y2)
            elif left and bottom:
                return dist(x1, y1, x2b, y2b)
            elif bottom and right:
                return dist(x1b, y1, x2, y2b)
            elif right and top:
                return dist(x1b, y1b, x2, y2)
            elif left:
                return x1 - x2b
            elif right:
                return x2 - x1b
            elif bottom:
                return y1 - y2b
            elif top:
                return y2 - y1b

            return 0.0

        adj = np.zeros((len(self.nodes), len(self.edge_labels), len(self.nodes)))

        max_x = np.max([n.x + n.w for n in self.nodes])
        max_y = np.max([n.y + n.h for n in self.nodes])
        min_x = np.min([n.x for n in self.nodes])
        min_y = np.min([n.y for n in self.nodes])

        max_delta_x = np.abs(max_x - min_x)
        max_delta_y = np.abs(max_y - min_y)

        # edgeType = 'fc_similarity' # 'fc_binary', 'fc_similarity', 'normal_binary'
        edgeType = "normal_binary"

        if "fc_similarity" in edgeType:
            # Fully connected

            for i in range(len(self.nodes)):
                for j in range(i, len(self.nodes)):
                    if i == j:
                        adj[i, :, j] = 1
                    else:
                        rect1 = scale_coor(self.nodes[i])
                        rect2 = scale_coor(self.nodes[j])
                        edist = np.abs(rect_distance(rect1, rect2))
                        adj[i, :, j] = (1 - (edist / np.sqrt(2))) ** 2
                        adj[j, :, i] = (1 - (edist / np.sqrt(2))) ** 2

                        # adj[i, :, j] = 1
                        # adj[j, :, i] = 1
            # ---------------------------------
        elif "fc_binary" in edgeType:
            # Fully connected

            for i in range(len(self.nodes)):
                for j in range(i, len(self.nodes)):
                    if i == j:
                        adj[i, :, j] = 1
                    else:
                        rect1 = scale_coor(self.nodes[i])
                        rect2 = scale_coor(self.nodes[j])
                        edist = np.abs(rect_distance(rect1, rect2))
                        adj[i, :, j] = 1
                        adj[j, :, i] = 1
            # ---------------------------------
        elif "normal_binary" in edgeType:
            for edge in self.edges:
                # start_center_x = edge.start.x + edge.start.w / 2
                # start_center_y = edge.start.y + edge.start.h / 2

                # end_center_x = edge.end.x + edge.end.w / 2
                # end_center_y = edge.end.y + edge.end.h / 2

                # delta_x = np.abs(start_center_x - end_center_x) / max_delta_x
                # delta_y = np.abs(start_center_y - end_center_y) / max_delta_y

                # edist = np.sqrt(delta_x ** 2 + delta_y ** 2)
                #
                # rect1 = scale_coor(edge.start)
                # rect2 = scale_coor(edge.end)
                # edist = np.abs(rect_distance(rect1, rect2))

                # Distance Euclidean distance based similarity is bounded in range [0, 1]
                # where 0 mean the two objects are to far from each other, and 1 means they have the same central gravity
                start = self.nodes.index(edge.start)
                end = self.nodes.index(edge.end)
                # adj[start, edge.label, end] = (1 - (edist / np.sqrt(2))) ** 2

                adj[start, edge.label, end] = 1

            #     #print(adj[start, edge.label, end])
        else:
            raise Exception("Invalid edge type: " + str(edgeType))

        self.adj = adj.astype(np.float16)


    def _draw_debug_image(self):

        def draw_rectangle(draw, coordinates, color, width=1):
            for i in range(width):
                rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
                rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
                draw.rectangle((rect_start, rect_end), outline=color)

        max_x = max(n.x + n.w for n in self.nodes)
        max_y = max(n.y + n.h for n in self.nodes)
        min_x = min(n.x for n in self.nodes)
        min_y = min(n.y for n in self.nodes)

        dw, dh = max_x - min_x, max_y - min_y
        pad = 20

        clone_img = Image.fromarray(np.ones((dh + 2 * pad, dw + 2 * pad, 3), dtype='uint8') * 255)
        draw = ImageDraw.Draw(clone_img)
        font = ImageFont.truetype(
            FONT_PATH,
            size=28,)
            #encoding='utf-8-sig')

        for node in self.nodes:
            if node in self.text_lines:
                vertex_type = 0
                color = 'green'
            elif node in self.table_cells:
                vertex_type = 1
                color = 'blue'
            elif node in self.cols:
                vertex_type = 2
                color = 'red'
            elif node in self.rows:
                vertex_type = 3
                color = 'orange'
            else:
                vertex_type = -1
                color = 'black'

            x, y, w, h = node.x, node.y, node.w, node.h
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1, x2, y2 = x1 - min_x + pad, y1 - min_y + pad, x2 - min_x + pad, y2 - min_y + pad
            text = node.get_text() if vertex_type in [0, 1] else ''
            text_color = 'red' if node.label > 0 else 'black'
            if node.label > 0:
                text = text + '  ({})'.format(node.label)
            draw.text((x1, y1), text, fill=text_color, font=font)
            draw_rectangle(draw, ((x1, y1), (x2, y2)), color=color, width=3)

        A_adj = self.adj
        N = len(self.nodes)
        tb = self.adj[:, 3, :]
        bt = self.adj[:, 2, :]
        rl = self.adj[:, 1, :]
        lr = self.adj[:, 0, :]

        for a_id in range(N):
            if not self.nodes[a_id] in self.table_cells:
                continue
            ax, ay = self.nodes[a_id].xc - min_x + pad, self.nodes[a_id].yc  - min_y + pad
            for b_id in range(N):
                if not self.nodes[b_id] in self.table_cells:
                    continue
                bx, by = self.nodes[b_id].xc - min_x + pad, self.nodes[b_id].yc - min_y + pad
                if tb[a_id][b_id] > 0 and bt[b_id][a_id] > 0:
                    draw.line(((ax + 1, ay + 1), (bx, by)), fill='red', width=3)
                # if bt[a_id][b_id] > 0:
                #     draw.line(((ax - 1, ay - 1), (bx, by)), fill='orange', width=3)
                if lr[a_id][b_id] > 0 and rl[b_id][a_id] > 0:
                    draw.line(((ax - 1, ay - 1), (bx, by)), fill='green', width=3)

        return clone_img