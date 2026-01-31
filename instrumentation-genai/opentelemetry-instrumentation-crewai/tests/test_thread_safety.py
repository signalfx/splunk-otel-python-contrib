"""Tests for thread safety in CrewAI instrumentation."""

import concurrent.futures
import threading
from unittest import mock

import opentelemetry.instrumentation.crewai.instrumentation as crewai_module
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor


class TestThreadSafety:
    """Test suite for thread safety of CrewAI instrumentation."""

    def test_concurrent_workflow_spans(self, stub_handler):
        """Test that concurrent workflow executions create separate spans."""
        crewai_module._handler = stub_handler

        def execute_workflow(workflow_id):
            """Simulate a workflow execution."""
            mock_crew = mock.MagicMock()
            mock_crew.name = f"Workflow-{workflow_id}"

            result = mock.MagicMock()
            result.raw = f"Result-{workflow_id}"
            mock_wrapped = mock.MagicMock(return_value=result)

            # Call the wrapper function
            crewai_module._wrap_crew_kickoff(
                mock_wrapped, mock_crew, (), {"inputs": {"id": workflow_id}}
            )

            return workflow_id

        # Execute multiple workflows concurrently
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(execute_workflow, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Wait for completion

        # Verify all workflows were started and stopped
        assert len(stub_handler.started_workflows) == num_threads
        assert len(stub_handler.stopped_workflows) == num_threads

        # Verify each workflow has a unique name
        workflow_names = [w.name for w in stub_handler.started_workflows]
        assert len(set(workflow_names)) == num_threads

    def test_concurrent_agent_executions(self, stub_handler):
        """Test that concurrent agent executions are properly tracked."""
        crewai_module._handler = stub_handler

        def execute_agent(agent_id):
            """Simulate an agent execution."""
            mock_agent = mock.MagicMock()
            mock_agent.role = f"Agent-{agent_id}"

            mock_task = mock.MagicMock()
            mock_task.description = f"Task for Agent-{agent_id}"

            mock_wrapped = mock.MagicMock(return_value=f"Agent-{agent_id} result")

            crewai_module._wrap_agent_execute_task(
                mock_wrapped, mock_agent, (), {"task": mock_task}
            )

            return agent_id

        # Execute multiple agents concurrently
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(execute_agent, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Wait for completion

        # Verify all agents were started and stopped
        assert len(stub_handler.started_agents) == num_threads
        assert len(stub_handler.stopped_agents) == num_threads

        # Verify each agent has a unique name
        agent_names = [a.name for a in stub_handler.started_agents]
        assert len(set(agent_names)) == num_threads

    def test_concurrent_task_executions(self, stub_handler):
        """Test that concurrent task executions are properly tracked."""
        crewai_module._handler = stub_handler

        def execute_task(task_id):
            """Simulate a task execution."""
            mock_task = mock.MagicMock()
            mock_task.description = f"Task-{task_id}"
            mock_task.expected_output = f"Expected output for Task-{task_id}"
            mock_task.agent = mock.MagicMock()
            mock_task.agent.role = f"Agent-{task_id}"

            mock_wrapped = mock.MagicMock(return_value=f"Task-{task_id} result")

            crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})

            return task_id

        # Execute multiple tasks concurrently
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(execute_task, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Wait for completion

        # Verify all tasks (steps) were started and stopped
        assert len(stub_handler.started_steps) == num_threads
        assert len(stub_handler.stopped_steps) == num_threads

        # Verify each task has a unique description
        step_names = [s.name for s in stub_handler.started_steps]
        assert len(set(step_names)) == num_threads

    def test_concurrent_tool_calls(self, stub_handler):
        """Test that concurrent tool calls are properly tracked."""
        crewai_module._handler = stub_handler

        def execute_tool(tool_id):
            """Simulate a tool execution."""
            mock_tool = mock.MagicMock()
            mock_tool.name = f"Tool-{tool_id}"

            mock_wrapped = mock.MagicMock(return_value=f"Tool-{tool_id} result")

            crewai_module._wrap_tool_call(
                mock_wrapped, mock_tool, (), {"param": f"value-{tool_id}"}
            )

            return tool_id

        # Execute multiple tools concurrently
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(execute_tool, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Wait for completion

        # Verify all tool calls were started and stopped
        assert len(stub_handler.started_tool_calls) == num_threads
        assert len(stub_handler.stopped_tool_calls) == num_threads

        # Verify each tool call has a unique name
        tool_names = [t.name for t in stub_handler.started_tool_calls]
        assert len(set(tool_names)) == num_threads

    def test_concurrent_mixed_operations(self, stub_handler):
        """Test that mixed concurrent operations (workflow, agent, task, tool) are properly tracked."""
        crewai_module._handler = stub_handler

        def execute_workflow(workflow_id):
            mock_crew = mock.MagicMock()
            mock_crew.name = f"Workflow-{workflow_id}"
            result = mock.MagicMock()
            result.raw = f"Result-{workflow_id}"
            mock_wrapped = mock.MagicMock(return_value=result)
            crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})
            return ("workflow", workflow_id)

        def execute_agent(agent_id):
            mock_agent = mock.MagicMock()
            mock_agent.role = f"Agent-{agent_id}"
            mock_task = mock.MagicMock()
            mock_task.description = f"Task-{agent_id}"
            mock_wrapped = mock.MagicMock(return_value="Agent result")
            crewai_module._wrap_agent_execute_task(
                mock_wrapped, mock_agent, (), {"task": mock_task}
            )
            return ("agent", agent_id)

        def execute_task(task_id):
            mock_task = mock.MagicMock()
            mock_task.description = f"Task-{task_id}"
            mock_task.expected_output = "Output"
            mock_task.agent = mock.MagicMock()
            mock_task.agent.role = "Agent"
            mock_wrapped = mock.MagicMock(return_value="Task result")
            crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})
            return ("task", task_id)

        def execute_tool(tool_id):
            mock_tool = mock.MagicMock()
            mock_tool.name = f"Tool-{tool_id}"
            mock_wrapped = mock.MagicMock(return_value="Tool result")
            crewai_module._wrap_tool_call(mock_wrapped, mock_tool, (), {})
            return ("tool", tool_id)

        # Execute mixed operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(execute_workflow, i))
                futures.append(executor.submit(execute_agent, i))
                futures.append(executor.submit(execute_task, i))
                futures.append(executor.submit(execute_tool, i))

            for f in concurrent.futures.as_completed(futures):
                f.result()  # Wait for completion

        # Verify all operations completed
        assert len(stub_handler.started_workflows) == 5
        assert len(stub_handler.stopped_workflows) == 5
        assert len(stub_handler.started_agents) == 5
        assert len(stub_handler.stopped_agents) == 5
        assert len(stub_handler.started_steps) == 5
        assert len(stub_handler.stopped_steps) == 5
        assert len(stub_handler.started_tool_calls) == 5
        assert len(stub_handler.stopped_tool_calls) == 5

    def test_concurrent_instrumentation_does_not_crash(
        self, tracer_provider, meter_provider
    ):
        """Test that concurrent instrumentation calls don't crash the application."""
        errors = []
        lock = threading.Lock()

        def instrument():
            try:
                instrumentor = CrewAIInstrumentor()
                with mock.patch(
                    "opentelemetry.instrumentation.crewai.instrumentation.wrap_function_wrapper"
                ):
                    instrumentor.instrument(
                        tracer_provider=tracer_provider,
                        meter_provider=meter_provider,
                    )
                return True
            except Exception as e:
                with lock:
                    errors.append(str(e))
                return False

        # Try to instrument from multiple threads simultaneously
        num_threads = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(instrument) for _ in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Wait for completion

        # No thread should have encountered an unhandled exception
        assert len(errors) == 0, (
            f"Errors occurred during concurrent instrumentation: {errors}"
        )

        # The handler should exist after instrumentation
        assert crewai_module._handler is not None

        # Cleanup - create a new instrumentor to uninstrument
        instrumentor = CrewAIInstrumentor()
        with mock.patch("opentelemetry.instrumentation.crewai.instrumentation.unwrap"):
            try:
                instrumentor.uninstrument()
            except Exception:
                pass  # Ignore cleanup errors

    def test_handler_access_during_concurrent_execution(self, stub_handler):
        """Test that handler access is safe during concurrent span operations."""
        crewai_module._handler = stub_handler

        barrier = threading.Barrier(4)

        def workflow_operation():
            barrier.wait()  # Sync all threads
            mock_crew = mock.MagicMock()
            mock_crew.name = "Concurrent Crew"
            result = mock.MagicMock()
            result.raw = "Result"
            mock_wrapped = mock.MagicMock(return_value=result)
            crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})
            return "workflow"

        def agent_operation():
            barrier.wait()  # Sync all threads
            mock_agent = mock.MagicMock()
            mock_agent.role = "Concurrent Agent"
            mock_task = mock.MagicMock()
            mock_task.description = "Task"
            mock_wrapped = mock.MagicMock(return_value="Agent result")
            crewai_module._wrap_agent_execute_task(
                mock_wrapped, mock_agent, (), {"task": mock_task}
            )
            return "agent"

        def task_operation():
            barrier.wait()  # Sync all threads
            mock_task = mock.MagicMock()
            mock_task.description = "Concurrent Task"
            mock_task.expected_output = "Output"
            mock_task.agent = mock.MagicMock()
            mock_task.agent.role = "Agent"
            mock_wrapped = mock.MagicMock(return_value="Task result")
            crewai_module._wrap_task_execute(mock_wrapped, mock_task, (), {})
            return "task"

        def tool_operation():
            barrier.wait()  # Sync all threads
            mock_tool = mock.MagicMock()
            mock_tool.name = "Concurrent Tool"
            mock_wrapped = mock.MagicMock(return_value="Tool result")
            crewai_module._wrap_tool_call(mock_wrapped, mock_tool, (), {})
            return "tool"

        # Execute all operations at the same time using barrier
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(workflow_operation),
                executor.submit(agent_operation),
                executor.submit(task_operation),
                executor.submit(tool_operation),
            ]
            results = [f.result() for f in futures]

        # Verify all operations completed
        assert set(results) == {"workflow", "agent", "task", "tool"}
        assert len(stub_handler.started_workflows) == 1
        assert len(stub_handler.started_agents) == 1
        assert len(stub_handler.started_steps) == 1
        assert len(stub_handler.started_tool_calls) == 1

    def test_error_handling_in_concurrent_operations(self, stub_handler):
        """Test that errors in concurrent operations are properly handled."""
        crewai_module._handler = stub_handler

        def execute_failing_workflow(workflow_id):
            """Simulate a failing workflow execution."""
            mock_crew = mock.MagicMock()
            mock_crew.name = f"Failing-Workflow-{workflow_id}"

            def raise_error(*args, **kwargs):
                raise ValueError(f"Workflow {workflow_id} failed")

            mock_wrapped = mock.MagicMock(side_effect=raise_error)

            try:
                crewai_module._wrap_crew_kickoff(mock_wrapped, mock_crew, (), {})
            except ValueError:
                pass

            return workflow_id

        # Execute multiple failing workflows concurrently
        num_threads = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(execute_failing_workflow, i) for i in range(num_threads)
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Wait for completion

        # Verify all workflows were started
        assert len(stub_handler.started_workflows) == num_threads
        # Verify all failures were recorded
        assert len(stub_handler.failed_entities) == num_threads
        # Verify no workflows were stopped (they all failed)
        assert len(stub_handler.stopped_workflows) == 0
