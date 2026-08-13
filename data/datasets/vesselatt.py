# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import pdb
import os.path as osp
import os
import json

from .bases import BaseImageDatasetAtt


class VesselAtt(BaseImageDatasetAtt):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'data' #'market1501'

    def __init__(self, root='/home/haoluo/data', att_root='/home/haoluo/data', verbose=True, **kwargs):
        super(VesselAtt, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.att_dir = att_root

        self._preprocess_attdata(self.att_dir)

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, withatt=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info_withatt(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info_withatt(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info_withatt(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _preprocess_attdata(self, att_root):
        att_paths = glob.glob(osp.join(att_root, '*.txt'))

        jsondata = self._load_json(att_paths[0])
        att_var = [set() for i in range(len(jsondata.keys()))]

        att_temp_data = []
        for att_path in att_paths:
            jsondata = self._load_json(att_path)
            att_temp_data.append(jsondata)
            for i, (k, v) in enumerate(jsondata.items()):
                att_var[i].add(v)
            ####
        ####

        self.att2label = []
        for att_v in att_var:
            self.att2label.append({id: label for label, id in enumerate(att_v)})
        ####


    def _load_json(self, filename):
        fstr = ""
        f = open(filename, "r")
        for ll in f:
            if ll[0] == '`':
                continue
            fstr += ll.strip()
        f.close()
        jsdata = json.loads(fstr) #load(open(ff, "r"))
        return jsdata

    def _process_dir(self, dir_path, relabel=False, withatt=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            attfn = (img_path.split("/")[-1]).split(".")[0] + ".txt"
            if withatt:
                attjson = self._load_json(os.path.join(self.att_dir, attfn))
                attlabels = []
                #pdb.set_trace()
                for i, (k, v) in enumerate(attjson.items()):
                    attlabels.append(self.att2label[i][v])
                ####
            ####
            #pdb.set_trace()
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            if withatt:
                dataset.append((img_path, pid, camid, attlabels))
            else:
                dataset.append((img_path, pid, camid, None))
            ####
        return dataset



class VesselAttV2(BaseImageDatasetAtt):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'data' #'market1501'

    def __init__(self, root='/home/haoluo/data', att_root='/home/haoluo/data', verbose=True, **kwargs):
        super(VesselAttV2, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.att_dir = att_root

        self._preprocess_attdata(self.att_dir)

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, withatt=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info_withatt(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info_withatt(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info_withatt(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _preprocess_attdata(self, att_root):
        att_paths = glob.glob(osp.join(att_root, '*.txt'))

        jsondata = self._load_json(att_paths[0])
        att_var = [set() for i in range(len(jsondata.keys()))]

        att_temp_data = []
        for att_path in att_paths:
            jsondata = self._load_json(att_path)
            att_temp_data.append(jsondata)
            for i, (k, v) in enumerate(jsondata.items()):
                att_var[i].add(v)
            ####
        ####

        self.att2label = []
        for att_v in att_var:
            self.att2label.append({id: label for label, id in enumerate(att_v)})
        ####


    def _load_json(self, filename):
        fstr = ""
        f = open(filename, "r")
        for ll in f:
            if ll[0] == '`':
                continue
            fstr += ll.strip()
        f.close()
        jsdata = json.loads(fstr) #load(open(ff, "r"))
        return jsdata

    def _process_dir(self, dir_path, relabel=False, withatt=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            attfn = (img_path.split("/")[-1]).split(".")[0] + ".txt"
            if withatt:
                attjson = self._load_json(os.path.join(self.att_dir, attfn))
                attlabels = []
                #pdb.set_trace()
                for i, (k, v) in enumerate(attjson.items()):
                    attlabels.append(self.att2label[i][v])
                ####
            ####
            #pdb.set_trace()
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 2400  # pid == 0 means background
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            if withatt:
                dataset.append((img_path, pid, camid, attlabels))
            else:
                dataset.append((img_path, pid, camid, None))
            ####
        return dataset


