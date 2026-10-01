"""Causal event replay tests; no training and no calibration energy access."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from energyville_evb02_data import replay_events,summary,SECOND


class EventReplayTests(unittest.TestCase):
    def test_permissions_release_and_future_mutation(self):
        records=[('a','BEV1','train',0,10),('cal','BEV1','calibration',12,20),('other','BEV2','train',0,10),
                 ('long','BEV1','train',5,100),('v','BEV1','validation',30,40),('d','BEV1','future',50,60),('q','BEV1','future',70,80)]
        meta=pd.DataFrame(records,columns=['file','vehicle','split','cut_ns','end_ns'])
        meta[['cut_ns','end_ns']]*=SECOND;meta['od_key']='x'
        for k in ('main_view_internal_consistency_checked',):meta[k]=True
        for k in ('uncertain_charger_signal_or_mode_overlap','confirmed_external_charging_from_counter_or_joint_direct'):meta[k]=False
        meta['overlap_conflicting_iv_rows']=0
        def run(mutation=0):
            called=[]
            def read(i):
                name=meta.loc[i,'file'];called.append(name)
                if name=='cal':raise AssertionError('calibration label consumed')
                return dict(energy_kwh=mutation if name=='long' else 2.,distance_km=10.,duration_s=100.,ambient_mean=12.,pack_min_mean=15.,pack_max_mean=16.)
            return replay_events(meta,read),called
        (a,provenance,snap,desc),called=run(100.)
        (b,_,_,_),_=run(-100.)
        pd.testing.assert_frame_equal(a,b)
        # A deterministic frozen readout's predictions also cannot change.
        np.testing.assert_array_equal(a.history_count.fillna(0)+a.history_net_wh_per_km.fillna(0),b.history_count.fillna(0)+b.history_net_wh_per_km.fillna(0))
        self.assertNotIn('long',called);self.assertNotIn('cal',called)
        self.assertEqual(a.loc[4,'history_count'],1);self.assertEqual(a.loc[5,'history_count'],2);self.assertEqual(a.loc[6,'history_count'],3)
        self.assertEqual(set(snap[-1]['latest10_ids']),{'a','v','d'})
        self.assertTrue(all(p['history_available_ns']<=p['cut_ns'] and p['history_file']!=p['query_file'] for p in provenance))

    def test_full_source_rate_and_signed_energy(self):
        prior=[dict(end_ns=10*SECOND,energy_kwh=-1.,distance_km=10.,duration_s=100.,ambient_mean=10.,pack_min_mean=15.,pack_max_mean=16.)]
        s=summary(prior,prior,12*SECOND)
        self.assertEqual(s['history_net_wh_per_km'],-100.)
        self.assertEqual(s['history_recent_age_s'],1.)
        self.assertEqual(s['history_mean_speed_kph'],360.)
        self.assertEqual(s['od_history_count'],1)

    def test_bucket_end_boundary(self):
        base=dict(vehicle='BEV1',split='train',od_key='x',main_view_internal_consistency_checked=True,uncertain_charger_signal_or_mode_overlap=False,confirmed_external_charging_from_counter_or_joint_direct=False,overlap_conflicting_iv_rows=0)
        m=pd.DataFrame([{**base,'file':'source','cut_ns':0,'end_ns':10*SECOND},{**base,'file':'early','cut_ns':10*SECOND,'end_ns':20*SECOND},{**base,'file':'eligible','cut_ns':11*SECOND,'end_ns':30*SECOND}])
        def reader(i):return dict(energy_kwh=1.,distance_km=1.,duration_s=10.,ambient_mean=10.,pack_min_mean=10.,pack_max_mean=11.)
        f,_,_,_=replay_events(m,reader)
        self.assertEqual(f.loc[1,'history_count'],0);self.assertEqual(f.loc[2,'history_count'],1)


if __name__=='__main__':unittest.main()
