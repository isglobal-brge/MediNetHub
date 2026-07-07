"""
C3 — Flower client authorization (invited-client allowlist).

Verifies that aggregation drops fit results from clients whose reported
client_id is not in the job's invited allowlist (poisoning / rogue-client
defense), and that with no allowlist configured the filter is a backward-
compatible no-op.
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')
os.environ.setdefault('DJANGO_DEBUG', 'True')
django.setup()

import unittest
from collections import namedtuple

from webapp.server_fn.strategies import (
    filter_authorized_clients,
    get_expected_client_ids,
)

_FitRes = namedtuple('_FitRes', ['metrics'])
_CP = namedtuple('_CP', ['cid'])
_Job = namedtuple('_Job', ['config_json', 'clients_config'])


class C3ClientAuthTests(unittest.TestCase):
    def _results(self):
        return [
            (_CP('x'), _FitRes({'client_id': 'tok-A'})),
            (_CP('y'), _FitRes({'client_id': 'ROGUE'})),
            (_CP('z'), _FitRes({'client_id': 'tok-B'})),
        ]

    def test_unauthorized_client_dropped(self):
        kept, dropped = filter_authorized_clients(self._results(), {'tok-A', 'tok-B'}, 1)
        self.assertEqual(dropped, 1)
        self.assertEqual({f.metrics['client_id'] for _, f in kept}, {'tok-A', 'tok-B'})

    def test_no_allowlist_is_noop(self):
        kept, dropped = filter_authorized_clients(self._results(), set(), 1)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 3)

    def test_all_unauthorized_dropped(self):
        kept, dropped = filter_authorized_clients(self._results(), {'other'}, 1)
        self.assertEqual(dropped, 3)
        self.assertEqual(kept, [])

    def test_expected_ids_from_clients_config(self):
        job = _Job({}, {'tok-A': {}, 'tok-B': {}})
        self.assertEqual(get_expected_client_ids(job), {'tok-A', 'tok-B'})

    def test_expected_ids_explicit_config_takes_precedence(self):
        job = _Job({'expected_client_ids': ['s1', 's2']}, {'ignored': {}})
        self.assertEqual(get_expected_client_ids(job), {'s1', 's2'})

    def test_expected_ids_empty_when_unconfigured(self):
        job = _Job({}, {})
        self.assertEqual(get_expected_client_ids(job), set())


if __name__ == '__main__':
    unittest.main()
