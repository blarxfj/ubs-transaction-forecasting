"""Small invariant tests; no challenge labels or held-out predictions are needed."""
import unittest
import numpy as np
import pandas as pd
from features import LABELS,hashed,transaction_features,assign_families,one_client,stats
from evaluate import confusion_f1
from run import adjust,final
from sklearn.metrics import f1_score


def events():
    return pd.DataFrame([
        dict(client_id='example',timestamp=t,amount=a,currency='chf',direction='out',type='card_payment',mcc=m,description=d,fee=0.)
        for t,a,m,d in [
            ('2025-10-12T12:00:00Z',9.9,5732,'cloud backup'),
            ('2025-11-12T12:00:00Z',10.,5732,'storage plan'),
            ('2025-12-12T12:00:00Z',10.1,5411,'subscription charge'),
            ('2025-12-20T12:00:00Z',50.,5411,'grocery store'),
        ]])

class Tests(unittest.TestCase):
    def test_repair_family(self):
        tx=transaction_features(events())
        f,_=assign_families(tx)
        np.testing.assert_array_equal(f,[0,0,0,-1])

    def test_future_rejected(self):
        tx=events();tx.loc[0,'timestamp']='2026-01-02T00:00:00Z'
        with self.assertRaises(AssertionError):transaction_features(tx)

    def test_order_invariant(self):
        tx=transaction_features(events());a=one_client(tx);b=one_client(tx.iloc[::-1])
        self.assertEqual(a[0],b[0])
        for fa,fb in zip(a[1],b[1]):
            self.assertEqual(set(fa),set(fb))
            for key in fa:np.testing.assert_allclose(fa[key],fb[key],equal_nan=True)

    def test_period(self):
        result=stats(np.array([-80.,-50.,-20.]),np.array([10.,10.1,10.]))
        self.assertEqual(result['cadence'],30.)
        self.assertAlmostEqual(result['cadence_due'],10.)

    def test_hash(self):
        self.assertEqual(hashed('example'),hashed('example'))
        self.assertTrue(0<=hashed('0:example')%5<5)
        self.assertNotEqual(hashed('0:example'),hashed('1:example'))

    def test_fixed_label_score(self):
        y=np.array([0,0,1,7]);p=np.array([0,1,1,7])
        self.assertAlmostEqual(confusion_f1(y,p).mean(),f1_score(y,p,average='macro',labels=range(8),zero_division=0))

    def test_lockbox_excluded_from_determinism_check(self):
        with self.assertRaises(ValueError):
            final(None,None,None,None,None,True,True)

    def test_probability_adjustment(self):
        p=adjust(np.full((2,8),1/8),{'none_weight':.8})
        np.testing.assert_allclose(p.sum(1),1)
        self.assertTrue(np.all(p[:,7]<p[:,0]))
        self.assertEqual(LABELS,['cloud','gym','insurance','mobile','music','software','streaming','none'])

if __name__=='__main__':unittest.main()
