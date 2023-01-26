import logging
from nats_bench import create
import pytest

import sys; sys.path.append('../src')

from utils.nas import index_to_nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_index_to_nx():
    """Ensure that mapping from index to arch is correct."""
    for api_index in range(15625):
        api = create(None, 'tss', fast_mode=True, verbose=False)
        arch = index_to_nx(api, api_index)
        if arch is not None:
            queried_index = api.query_index_by_arch(arch.name)
            logger.info(f'{api_index} == {queried_index}')
            assert api_index == queried_index
