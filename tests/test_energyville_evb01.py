"""Small synthetic causal/protocol tests; never fit a model locally."""
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from energyville_evb01_data import geometry_features,latest_state,ns,temporal_split,historical_features,TREE_CONFIG,SECOND
from run_energyville_evb01 import scores,simple_choice,paired_bootstrap


class EVB01Tests(unittest.TestCase):
    def test_geometry_repeated_sampling_invariant(self):
        la=np.array([51.,51.001,51.002,51.003]);lo=np.array([4.,4.001,4.001,4.002])
        a,_=geometry_features(la,lo,1.);b,_=geometry_features(np.repeat(la,[30,1,200,2]),np.repeat(lo,[30,1,200,2]),1.)
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-12)

    def test_ns_units_and_future_state_exclusion(self):
        time=ns(pd.Series(['2025-01-01T00:00:00Z','2025-01-01T00:00:01Z','2025-01-01T00:00:02Z']))
        self.assertEqual(time[1]-time[0],SECOND)
        d=pd.DataFrame(dict(SOCave292=[80,79,1],VCRIGHT_tempAmbientRaw=[10,11,80],VCFRONT_tempAmbient=[9,10,70],BMSminPackTemperature=[20,21,90],BMSmaxPackTemperature=[22,23,95],UI_batteryPreconditioningRequest=[0,1,0]))
        a,p=latest_state(d,time,time[1],'toy');self.assertEqual(a['soc'],80);self.assertEqual(a['pack_min_c'],20)
        d.loc[1:,:]=np.nan;b,_=latest_state(d,time,time[1],'toy')
        for key in a:np.testing.assert_allclose(a[key],b[key],equal_nan=True)
        self.assertTrue(all(x['available_ns']<=time[1] for x in p))

    def test_date_split_and_purge(self):
        cut=pd.date_range('2025-01-01',periods=100,tz='UTC');times=cut.as_unit('ns').asi8
        c=pd.DataFrame(dict(file=[str(i) for i in range(100)],t_cut=cut,cut_ns=times,end_ns=times+SECOND))
        c.loc[59,'end_ns']=times[60]+SECOND
        split,contract=temporal_split(c)
        self.assertEqual(contract['boundary_dates'],['2025-03-02','2025-03-17','2025-03-27']);self.assertEqual(split[59],'purged')
        self.assertEqual(split[60],'validation');self.assertEqual(len(set(np.flatnonzero(split=='train'))&set(np.flatnonzero(split=='future'))),0)

    def test_training_history_completion_and_self_exclusion(self):
        c=pd.DataFrame(dict(file=['a','b','c'],vehicle=['v']*3,split=['train','train','future'],start_ns=[0,10*SECOND,20*SECOND],end_ns=[5*SECOND,15*SECOND,25*SECOND],cut_ns=[SECOND,11*SECOND,21*SECOND]))
        route=pd.DataFrame(dict(od_key=['x']*3));desc=pd.DataFrame(dict(energy_kwh=[1.,2.,999.],distance_km=[10.,10.,10.],duration_s=[5.]*3,ambient_mean=[10.]*3,pack_min_mean=[20.]*3,pack_max_mean=[22.]*3))
        th={'v':pd.DataFrame(dict(timestamp_ns=[-SECOND],min_temperature=[19.],max_temperature=[21.],file=['past'],mode=['park']))}
        h,p,t=historical_features(c,route,c.loc[c.split=='train'],desc,th)
        self.assertEqual(h.history_count.tolist(),[0,1,2]);self.assertEqual(h.history_net_wh_per_km.iloc[2],150.)
        desc.loc[2,'energy_kwh']=-1e8;h2,_,_=historical_features(c,route,c.loc[c.split=='train'],desc,th);pd.testing.assert_frame_equal(h,h2)
        self.assertTrue(all(x['history_file']!=x['file'] and x['history_available_ns']<=x['cut_ns'] for x in p))
        self.assertTrue(all(x['available_ns']<=x['cut_ns'] for x in t))

    def test_metrics_tie_and_frozen_config(self):
        y=np.array([-.1,.05,1.]);p=np.array([0.,.1,.8]);v=np.array(['a','a','b']);a=scores(y,p,v)
        self.assertEqual(a['mape_n'],1);self.assertAlmostEqual(a['mae_kwh'],(.1+.05+.2)/3)
        b=scores(y+2,p+2,v);self.assertAlmostEqual(a['mae_kwh'],b['mae_kwh'])
        self.assertEqual(simple_choice({'R':{'mae_kwh':1.005},'S':{'mae_kwh':1.}},'R','S'),'R')
        self.assertFalse(TREE_CONFIG['early_stopping']);self.assertEqual(TREE_CONFIG['max_iter'],500)

    def test_paired_week_short_sample_no_ci(self):
        c=pd.DataFrame(dict(t_cut=['2025-01-01T00:00:00Z','2025-01-02T00:00:00Z'],y_rem=[1.,2.]))
        for g in ('B0','R','S','H','O'):c['pred_'+g+'_rem']=[1.,2.]
        p=paired_bootstrap(c,c,dict(reference='B0',legal_candidate='S',chosen_legal='B0'))
        self.assertTrue(p.ci95_low_kwh.isna().all());self.assertTrue((p.week_blocks==1).all());self.assertTrue((p.mae_improvement_right_kwh==0).all())


if __name__=='__main__':unittest.main(verbosity=2)
