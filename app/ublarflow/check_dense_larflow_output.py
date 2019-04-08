from __future__ import print_function
import ROOT as rt
from ROOT import std
from larcv import larcv
from ublarcvapp import ublarcvapp
import sys
sys.argv.append("-b")
larcv.load_rootutil()

rt.gStyle.SetOptStat(0)

dense_lffile = sys.argv[1]

VIS_FLOWS = True
flow_dirs = ['y2u','y2v']
vis_check =  False
has_visi  = False

io = larcv.IOManager(larcv.IOManager.kREAD)
io.add_in_file( dense_lffile )
io.initialize()


nentries = io.get_n_entries()
print("Num Entries: ",nentries)
nentries = 1

thresholds_v = std.vector("float")(3,10.0)

if VIS_FLOWS:
    cflow = rt.TCanvas("cflow","cflow",1200,900)
    cflow.Divide(3,3)
    cflow.Draw()

for n in range(nentries):
    io.read_entry(n)

    # crop lists
    ev_adc = io.get_data(larcv.kProductImage2D,"adc")
    adc_v  = ev_adc.Image2DArray()
    print("number of adc crops: {} ({} per 3-plane set)".format(adc_v.size(),adc_v.size()/3))
    
    ev_flow = {}
    flow_v  = {}
    for flowdir in flow_dirs:
        ev_flow[flowdir] = io.get_data(larcv.kProductImage2D,"larflow_{}".format(flowdir))
        flow_v[flowdir]  = ev_flow[flowdir].Image2DArray()
        print("number of flow[{}]: {}".format(flowdir,flow_v[flowdir].size()))

    nimgsets = flow_v["y2u"].size()

    flow_vv = []
    for i in xrange(nimgsets):

        cropped_adc_v  = std.vector("larcv::Image2D")()
        cropped_flow_v = std.vector("larcv::Image2D")()
        cropped_visi_v = std.vector("larcv::Image2D")()

        for flowdir in flow_dirs:
            cropped_flow_v.push_back( flow_v[flowdir].at( i ) )

        for p in xrange(3):
            cropped_adc_v.push_back( adc_v.at( 3*i + p ) )
        
        hvis = std.vector("TH2D")()
        #lfcrop_algo.check_cropped_images( 2, cropped_adc_v, ev_status,
        #                                  thresholds_v, cropped_flow_v, cropped_visi_v, hvis, has_visi, vis_check )
        
        flow_vv.append( cropped_flow_v )
        if VIS_FLOWS:
            # first row and column: ADC images
            h_v = {}
            canvspot = [1,2,0]
            htitle = ["TARGET1","TARGET2","SOURCE"]
            for j in range(3):
                cflow.cd(canvspot[j]+1)
                h_v[j] = larcv.as_th2d( cropped_adc_v.at(j), "test%d_%d"%(i,j) )
                h_v[j].GetZaxis().SetRangeUser(-10,250)
                h_v[j].SetTitle("%s"%(htitle[j]))
                h_v[j].Draw("COLZ")                
            # first col as well for matrix
            cflow.cd(3+1)
            h_v[2].Draw("COLZ")
            cflow.cd(6+1)
            h_v[2].Draw("COLZ")

            # second row, the flows
            for j in xrange(0,2):
                cflow.cd(3+j+2)
                h_v[3+j] = larcv.as_th2d( cropped_flow_v.at(j), "flow%d_%d"%(i,j) )
                h_v[3+j].GetZaxis().SetRangeUser(-832,832)                
                h_v[3+j].Draw("COLZ")

            # third row, errors
            if vis_check:
                print("number of hists: %d"%(hvis.size()))
                for j in xrange(0,2):
                    cflow.cd(6+j+2)
                    hvis.at(2*j+1).Draw("COLZ")
                
            cflow.Update()
            print("Visualized adc and flow images")
            raw_input()
            
        
    # for i in xrange(out_v.size()):
    #     detsplit.Append(out_v.at(i))
    # out.set_id( io.event_id().run(), io.event_id().subrun(), io.event_id().event() )
    #print "save entry"
    #out.save_entry()
    break


io.finalize()

print("==========================================")


print("FIN")
raw_input()
