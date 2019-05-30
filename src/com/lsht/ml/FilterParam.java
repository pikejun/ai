package com.lsht.ml;

/**
 * Created by Junson on 2017/8/21.
 */
public class FilterParam
{
    private int filterCnt;
    private int filterDeep;
    private int filterHeight;
    private int filterWidth;

    public FilterParam(int filterCnt,int filterDeep,int filterHeight,int filterWidth)
    {
        this.filterCnt=filterCnt;
        this.filterDeep=filterDeep;
        this.filterHeight=filterHeight;
        this.filterWidth=filterWidth;
    }

    public int getFilterCnt()
    {
        return filterCnt;
    }

    public void setFilterCnt(int filterCnt)
    {
        this.filterCnt = filterCnt;
    }

    public int getFilterDeep()
    {
        return filterDeep;
    }

    public void setFilterDeep(int filterDeep)
    {
        this.filterDeep = filterDeep;
    }

    public int getFilterHeight()
    {
        return filterHeight;
    }

    public void setFilterHeight(int filterHeight)
    {
        this.filterHeight = filterHeight;
    }

    public int getFilterWidth()
    {
        return filterWidth;
    }

    public void setFilterWidth(int filterWidth)
    {
        this.filterWidth = filterWidth;
    }
}
