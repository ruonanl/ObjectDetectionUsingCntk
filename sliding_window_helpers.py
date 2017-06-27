import csv, numpy

def valid_roi (imW, imH, x1, y1, x2, y2):
    if x2<=x1 or y2<=y1:
        return False
    if x1<0 or x2>=imW:
        return False
    if y1<0 or y2>=imH:
        return False
    return True


def get_grid_rois(imW, imH, stepsize, rectangles):
    rois = []
    for rec in rectangles:
        x1 = 0
        w = rec[0]
        h = rec[1]
        while x1 + w - 1 < imW:
            y1 = 0
            while y1 + h -1 < imH:
                if valid_roi (imW, imH, x1, y1, x1 + w -1, y1+h-1):
                    rois.append([x1, y1, x1 + w -1, y1+h-1])
                y1 = y1 + stepsize
            x1 = x1 + stepsize
    return rois
                

def find_tight_roi_from_GT_mast(bboxpath,labelpath, rectangles, scaling,imW, inH):

    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    label=[]
    
    bboxlines = csv.reader(open(bboxpath, newline='\n'), delimiter='\t')
    labellines = csv.reader(open(labelpath, newline='\n'), delimiter='\t')
    for r in bboxlines:
        xmin.append(int(r[0]))
        xmax.append(int(r[2]))
        ymin.append(int(r[1]))
        ymax.append(int(r[3]))
    for r in labellines:
        label.append(r[0])

    xmin=numpy.array([xmin[x] for x in range(len(label)) if label[x]=='mast'])
    xmax=numpy.array([xmax[x] for x in range(len(label)) if label[x]=='mast'])
    ymin=numpy.array([ymin[x] for x in range(len(label)) if label[x]=='mast'])
    ymax=numpy.array([ymax[x] for x in range(len(label)) if label[x]=='mast'])

    CW = (xmin+xmax)/2
    CH = (ymin+ymax)/2
    width = xmax - xmin
    height = ymax -ymin
    tightRois = []

    for lat in range(len(width)):
        
        if width[lat]>height[lat]:
            SQ = width[lat]
        else:
            SQ = height[lat]

        for rec in rectangles:
            if rec[1]>=SQ:
               SQB = rec[1]
               break
        if SQ>=rectangles[-1][1]:
            SQB = imW

        scalingstepsize = (SQB-SQ)/scaling
        for sca in range(scaling+1):
            SQ_step = SQ + sca * scalingstepsize
            XMIN = CW[lat] - SQ_step/2
            XMAX = CW[lat] + SQ_step/2
            YMIN = CH[lat] - SQ_step/2
            YMAX = CH[lat] + SQ_step/2
            tightRois.append([XMIN, YMIN, XMAX, YMAX])

    return tightRois

def generate_rois(imW,imH,tightRois,shift,stepsize):
    generatedRois =[]
    shiftarray = numpy.arange(-stepsize,stepsize,shift)
    for roi in tightRois:
        for x in shiftarray:
            for y in shiftarray:
                xl = roi[0]+x
                xh = roi[2]+x
                yl = roi[1]+y
                yh = roi[3]+y
                if valid_roi(imW, imH, xl, yl, xh, yh):
                    generatedRois.append([xl, yl, xh, yh])
    return generatedRois

###########################################

def find_enclosing_roi_from_GT_mast(bboxpath,labelpath, rectangles, scaling,imW, inH):

    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    label=[]
    
    bboxlines = csv.reader(open(bboxpath, newline='\n'), delimiter='\t')
    labellines = csv.reader(open(labelpath, newline='\n'), delimiter='\t')
    for r in bboxlines:
        xmin.append(int(r[0]))
        xmax.append(int(r[2]))
        ymin.append(int(r[1]))
        ymax.append(int(r[3]))
    for r in labellines:
        label.append(r[0])

    xmin=min(numpy.array([xmin[x] for x in range(len(label)) if label[x]=='insulator']))
    xmax=max(numpy.array([xmax[x] for x in range(len(label)) if label[x]=='insulator']))
    ymin=min(numpy.array([ymin[x] for x in range(len(label)) if label[x]=='insulator']))
    ymax=max(numpy.array([ymax[x] for x in range(len(label)) if label[x]=='insulator']))

    CW = (xmin+xmax)/2
    CH = (ymin+ymax)/2
    width = xmax - xmin
    height = ymax -ymin
    tightRois = []

        
    if width>height:
        SQ = width
    else:
        SQ = height

    for rec in rectangles:
        if rec[1]>=SQ:
           SQB = rec[1]
           break
    if SQ>=rectangles[-1][1]:
        SQB = imW

    scalingstepsize = (SQB-SQ)/scaling
    for sca in range(scaling+1):
        SQ_step = SQ + sca * scalingstepsize
        XMIN = CW - SQ_step/2
        XMAX = CW + SQ_step/2
        YMIN = CH - SQ_step/2
        YMAX = CH + SQ_step/2
        tightRois.append([XMIN, YMIN, XMAX, YMAX])

    return tightRois

def get_grid_rois_inbox(tightRois, ratios, stepsize_ratio, imW, imH):
    rois = []
    for rec in tightRois:
        for ratio in ratios:
            xl = rec[0]
            yl = rec[1]
            xh = rec[2]
            yh = rec[3]
            lat = (yh-yl)*ratio
            stepsize = lat*stepsize_ratio
            while xl + lat <= xh:
                yl = rec[1]
                while yl + lat <= yh:
                    if valid_roi (imW, imH, xl, yl, xl + lat -1, yl + lat -1):
                        rois.append([xl, yl, xl + lat -1, yl + lat -1])
                    yl = yl + stepsize
                xl = xl + stepsize
    return rois


def get_grid_rois_inbox_enlarge(tightRois, enlarge, ratios, stepsize_ratio, imW, imH):
    rois = []
    for rec in tightRois:
        for ratio in ratios:
            xl = rec[0]
            yl = rec[1]
            xh = rec[2]
            yh = rec[3]
            lat = (yh-yl)*ratio
            stepsize = lat*stepsize_ratio
            xl = xl-enlarge
            yl = yl-enlarge
            xh = xh+enlarge
            yh = yh+enlarge
            
            while xl + lat <= xh:
                yl = rec[1]
                while yl + lat <= yh:
                    if valid_roi (imW, imH, xl, yl, xl + lat -1, yl + lat -1):
                        rois.append([xl, yl, xl + lat -1, yl + lat -1])
                    yl = yl + stepsize
                xl = xl + stepsize
    return rois
            

def find_tight_roi_from_GT_inbox(bboxpath,labelpath, ratios, scaling_inbox, tightRois):

    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    label=[]
	
    bboxlines = csv.reader(open(bboxpath, newline='\n'), delimiter='\t')
    labellines = csv.reader(open(labelpath, newline='\n'), delimiter='\t')
    for r in bboxlines:
        xmin.append(int(r[0]))
        xmax.append(int(r[2]))
        ymin.append(int(r[1]))
        ymax.append(int(r[3]))
    for r in labellines:
        label.append(r[0])

    xmin=numpy.array([xmin[x] for x in range(len(label)) if label[x]=='insulator'])
    xmax=numpy.array([xmax[x] for x in range(len(label)) if label[x]=='insulator'])
    ymin=numpy.array([ymin[x] for x in range(len(label)) if label[x]=='insulator'])
    ymax=numpy.array([ymax[x] for x in range(len(label)) if label[x]=='insulator'])

    xcenter = (xmin+xmax)/2
    ycenter = (ymin+ymax)/2
    width = xmax-xmin
    height = ymax -ymin

    rois = []


    for j in range(len(width)):

        if width[j]>height[j]:
            sq = width[j]/2
        else:
            sq = height[j]/2

        for roi in tightRois:

            for r in ratios:
                tightheight = (roi[3]-roi[1])*r
                if sq*2 < tightheight:
                    sqb = tightheight/2
                    break
            if sq*2 >= (roi[3]-roi[1])*ratios[-1]:
                sqb = (roi[3]-roi[1])*0.5
            scalingstepsize = (sqb-sq)/scaling_inbox


            for sca in range(scaling_inbox+1):
                sq_step = sq + sca * scalingstepsize
                xmin_n = xcenter[j] - sq_step
                xmax_n = xcenter[j] + sq_step
                ymin_n = ycenter[j] - sq_step
                ymax_n = ycenter[j] + sq_step
                rois.append([xmin_n, ymin_n, xmax_n, ymax_n])

    return rois

def generate_rois_inbox(imW,imH,tightRois,shift_ratio,stepsize_ratio):
    generatedRois =[]
    shiftarray = numpy.arange(-stepsize_ratio,stepsize_ratio,shift_ratio)
    for roi in tightRois:
        lat = roi[2]-roi[0]
        for x in shiftarray:
            for y in shiftarray:
                xl = roi[0]+x*lat
                xh = roi[2]+x*lat
                yl = roi[1]+y*lat
                yh = roi[3]+y*lat
                if valid_roi(imW, imH, xl, yl, xh, yh):
                    generatedRois.append([xl, yl, xh, yh])
    return generatedRois
