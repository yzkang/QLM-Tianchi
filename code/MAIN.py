#!/usr/bin/python
#  -*- coding: utf-8 -*-
# date: 2018
# author: Kang Yan Zhe
# desc: 千里马 风险识别算法竞赛

from XGB_LGB import xgb_lgb_cv_modeling
from GBDT_LGB import gbdt_lgb_cv_modeling

if __name__ == '__main__':

    xgb_lgb_cv_modeling()

    gbdt_lgb_cv_modeling()