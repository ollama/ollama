import pytest
from client import Ollama


model_id = 'phi'


@pytest.mark.asyncio
async def test_show():
    o = Ollama()
    info = await o.verify_running(model_id)
    assert info['details']['family'] == 'phi2'