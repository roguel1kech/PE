from agent.agent import agent_process

def test_agent_response():
    text = "Мне грустно"
    result = agent_process(text)
    assert isinstance(result, str)
    assert len(result) > 0
