import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.clean_channel_id_mapper import CleanChannelIdMapper
from data_juicer.ops.mapper.clean_id_card_mapper import CleanIdCardMapper
from data_juicer.ops.mapper.clean_jwt_mapper import CleanJwtMapper
from data_juicer.ops.mapper.clean_mac_mapper import CleanMacMapper
from data_juicer.ops.mapper.clean_path_mapper import CleanPathMapper
from data_juicer.ops.mapper.clean_pem_mapper import CleanPemMapper
from data_juicer.ops.mapper.clean_phone_mapper import CleanPhoneMapper
from data_juicer.ops.mapper.clean_secret_mapper import CleanSecretMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AtomicPiiMapperTest(DataJuicerTestCaseBase):

    def _run(self, op, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_clean_path_mapper(self):
        samples = [
            {'text': 'open /Users/alice/secret.txt now', 'target': 'open  now'},
            {'text': r'path C:\Users\alice\secret.txt done', 'target': 'path  done'},
        ]
        self._run(CleanPathMapper(), samples)

    def test_clean_phone_mapper(self):
        samples = [
            {'text': 'call 13812345678 asap', 'target': 'call  asap'},
            {'text': 'hotline +86 1234567890 now', 'target': 'hotline  now'},
        ]
        self._run(CleanPhoneMapper(), samples)

    def test_clean_id_card_mapper(self):
        samples = [
            {'text': 'id 11010519491231002X ok', 'target': 'id  ok'},
        ]
        self._run(CleanIdCardMapper(), samples)

    def test_clean_secret_mapper(self):
        samples = [
            {'text': 'api_key: SECRET123', 'target': 'api_key: '},
            {'text': 'token=abcdefg', 'target': 'token='},
        ]
        self._run(CleanSecretMapper(), samples)

    def test_clean_channel_id_mapper(self):
        samples = [
            {'text': 'channel: feishu', 'target': 'channel: '},
            {'text': 'open id ou_1234567890abcdef1234567890abcdef', 'target': 'open id '},
        ]
        self._run(CleanChannelIdMapper(), samples)

    def test_clean_jwt_mapper(self):
        samples = [
            {'text': 'jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig ok', 'target': 'jwt  ok'},
        ]
        self._run(CleanJwtMapper(), samples)

    def test_clean_pem_mapper(self):
        samples = [
            {
                'text': '-----BEGIN PRIVATE KEY-----\nMIIE\n-----END PRIVATE KEY-----',
                'target': '',
            },
        ]
        self._run(CleanPemMapper(), samples)

    def test_clean_mac_mapper(self):
        samples = [
            {'text': 'mac aa:bb:cc:dd:ee:ff done', 'target': 'mac  done'},
        ]
        self._run(CleanMacMapper(), samples)


if __name__ == '__main__':
    unittest.main()
