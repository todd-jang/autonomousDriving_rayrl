"""Garmin GPX → Isaac Sim USD world (road mesh + lane semantics)."""
import math,argparse,logging
import numpy as np
from pathlib import Path

log=logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")

WGS84_A=6378137.0; WGS84_E2=6.694379990141e-3
GPX_NS="http://www.topografix.com/GPX/1/1"

def ll_to_utm(lat,lon):
    import math
    lat_r=math.radians(lat); lon_r=math.radians(lon)
    zone=int((lon+180)/6)+1; lon0=math.radians((zone-1)*6-180+3)
    N=WGS84_A/math.sqrt(1-WGS84_E2*math.sin(lat_r)**2)
    A_=math.cos(lat_r)*(lon_r-lon0)
    M=WGS84_A*((1-WGS84_E2/4)*lat_r-(3*WGS84_E2/8)*math.sin(2*lat_r))
    k0=0.9996
    e=k0*N*(A_+(1-math.tan(lat_r)**2)*A_**3/6)+500000.0
    n=k0*(M+N*math.tan(lat_r)*A_**2/2)
    if lat<0: n+=10000000.0
    return e,n

def parse_gpx(path):
    import xml.etree.ElementTree as ET
    tree=ET.parse(path); root=tree.getroot(); pts=[]
    for tag in [f".//{{{GPX_NS}}}rtept",f".//{{{GPX_NS}}}trkpt"]:
        for pt in root.findall(tag):
            e_el=pt.find(f"{{{GPX_NS}}}ele")
            pts.append((float(pt.get("lat")),float(pt.get("lon")),
                        float(e_el.text) if e_el is not None else 0.0))
        if pts: break
    if not pts:
        log.warning("GPX 없음 → 워커힐 mock")
        xs=np.linspace(0,800,100); ys=np.sin(xs/40)*4.5; zs=xs*0.035
        return list(zip(xs,ys,zs)), True
    e0,n0=ll_to_utm(pts[0][0],pts[0][1])
    result=[]
    for lat,lon,ele in pts:
        e,n=ll_to_utm(lat,lon); result.append((e-e0,n-n0,ele))
    return result,False

def build_usd(pts,out_path,lane_width=3.5):
    try:
        from pxr import Usd,UsdGeom,Gf,UsdShade,Sdf
        stage=Usd.Stage.CreateNew(out_path)
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        world=UsdGeom.Xform.Define(stage,"/World")
        stage.SetDefaultPrim(world.GetPrim())
        # road mesh
        verts=[]; faces=[]
        for i,(x,y,z) in enumerate(pts):
            verts+=[(x-lane_width,y,z),(x+lane_width,y,z)]
            if i>0:
                b=2*(i-1); faces+=[[b,b+1,b+3],[b,b+3,b+2]]
        mesh=UsdGeom.Mesh.Define(stage,"/World/Road/Surface")
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in verts])
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        # lane semantics
        for name,sign,ltype in [("EgoLane",-1,"ego"),("OncomingLane",+1,"oncoming")]:
            lm=UsdGeom.Mesh.Define(stage,f"/World/LaneMeshes/{name}")
            lv=[(x+sign*lane_width*0.5-1,y,z) for x,y,z in pts]+\
               [(x+sign*lane_width*0.5+1,y,z) for x,y,z in pts]
            lm.CreatePointsAttr([Gf.Vec3f(*v) for v in lv])
            n=len(pts)
            lf=[[i,i+n,i+n+1] for i in range(n-1)]+[[i,i+n+1,i+1] for i in range(n-1)]
            lm.CreateFaceVertexCountsAttr([3]*len(lf))
            lm.CreateFaceVertexIndicesAttr([i for f in lf for i in f])
            p=lm.GetPrim()
            p.SetCustomDataByKey("lane_type",ltype)
            p.SetCustomDataByKey("lane_dir_x",1.0 if ltype=="ego" else -1.0)
            p.SetCustomDataByKey("lane_dir_y",0.0)
            p.SetCustomDataByKey("road_id",0)
        stage.GetRootLayer().Save()
        log.info("USD 저장: %s (%d pts)",out_path,len(pts))
    except ImportError:
        log.warning("pxr 없음 → JSON mock 저장")
        import json
        Path(out_path).write_text(json.dumps({"n_pts":len(pts),"mock":True},indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpx",default="maps/walkerhill.gpx")
    ap.add_argument("--out",default="maps/walkerhill_world.usd")
    ap.add_argument("--lane-width",type=float,default=3.5)
    args=ap.parse_args()
    Path("maps").mkdir(exist_ok=True)
    pts,is_mock=parse_gpx(args.gpx)
    build_usd(pts,args.out,args.lane_width)
    log.info("완료%s"," (mock)" if is_mock else "")

if __name__=="__main__":
    main()
