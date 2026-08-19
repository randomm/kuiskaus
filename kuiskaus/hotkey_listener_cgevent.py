import queue
import threading
from collections.abc import Callable
import Quartz


class HotkeyListenerCGEvent:
    def __init__(self, on_press: Callable[[], None], on_release: Callable[[], None]):
        """
        Initialize hotkey listener using CGEventTap

        Args:
            on_press: Function to call when hotkey is pressed
            on_release: Function to call when hotkey is released
        """
        self.on_press = on_press
        self.on_release = on_release

        # Track state
        self.is_pressed = False
        self.tap = None
        self.run_loop_source = None
        self.running = False

        # Single-threaded dispatcher: the event tap only enqueues callbacks;
        # a dedicated worker runs them in event order so a release always
        # observes the state of its press (issue #17).
        self._callback_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._worker: threading.Thread | None = None

    def _check_modifiers(self, flags: int) -> bool:
        """Check if the required modifier keys are pressed"""
        # CGEventFlags values
        kCGEventFlagMaskControl = 1 << 18
        kCGEventFlagMaskAlternate = 1 << 19  # Option key
        kCGEventFlagMaskCommand = 1 << 20

        # Check Control key
        has_control = bool(flags & kCGEventFlagMaskControl)
        # Check Option key
        has_option = bool(flags & kCGEventFlagMaskAlternate)
        # Check that Command is NOT pressed (to avoid conflicts)
        has_command = bool(flags & kCGEventFlagMaskCommand)

        return has_control and has_option and not has_command

    def _event_tap_callback(self, proxy, type_, event, refcon):
        """CGEventTap callback"""
        try:
            # Check if it's a flags changed event
            if type_ == Quartz.kCGEventFlagsChanged:
                flags = Quartz.CGEventGetFlags(event)
                modifiers_pressed = self._check_modifiers(flags)

                # Debug output
                if flags != 0:
                    print(
                        f"[DEBUG CGEvent] Modifier flags: {flags}, Control+Option pressed: {modifiers_pressed}"
                    )

                if modifiers_pressed and not self.is_pressed:
                    # Hotkey pressed
                    print("[DEBUG CGEvent] Hotkey pressed!")
                    self.is_pressed = True
                    if self.on_press:
                        # Enqueue for the single-threaded dispatcher so the
                        # callback runs in event order, never inline and never
                        # on a raw per-callback thread (issue #17).
                        self._dispatch(self.on_press)

                elif not modifiers_pressed and self.is_pressed:
                    # Hotkey released
                    print("[DEBUG CGEvent] Hotkey released!")
                    self.is_pressed = False
                    if self.on_release:
                        # Enqueue for the single-threaded dispatcher.
                        self._dispatch(self.on_release)

        except Exception as e:
            print(f"Error in event tap callback: {e}")

        # Return the event to continue processing
        return event

    def _dispatch(self, callback: Callable[[], None]):
        """Enqueue a callback for the single-threaded dispatcher."""
        self._callback_queue.put(callback)

    def _dispatcher_loop(self):
        """Worker loop: run callbacks strictly in event order."""
        while True:
            callback = self._callback_queue.get()
            if callback is None:  # Sentinel: shut down
                break
            try:
                callback()
            except Exception as e:
                print(f"Error in hotkey callback: {e}")
            finally:
                self._callback_queue.task_done()

    def _drain_for_tests(self):
        """Block until the dispatcher has run every queued callback (tests)."""
        self._callback_queue.join()

    def start(self):
        """Start listening for hotkeys"""
        if not self.running:
            self.running = True

            # Check accessibility permissions
            from ApplicationServices import AXIsProcessTrusted

            is_trusted = AXIsProcessTrusted()

            if not is_trusted:
                print("Accessibility permissions required!")
                print(
                    "Please grant accessibility permissions in System Preferences > Security & Privacy > Accessibility"
                )
                return False

            print("[DEBUG CGEvent] Creating CGEventTap...")

            # Start the single-threaded dispatcher worker (once).
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._dispatcher_loop,
                    daemon=True,
                    name="hotkey-cgevent-dispatcher",
                )
                self._worker.start()

            # Create event tap
            self.tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,  # Session level
                Quartz.kCGHeadInsertEventTap,  # Insert at head
                Quartz.kCGEventTapOptionListenOnly,  # Just listen, don't modify
                Quartz.CGEventMaskBit(
                    Quartz.kCGEventFlagsChanged
                ),  # Monitor modifier key changes
                self._event_tap_callback,
                None,  # refcon
            )

            if not self.tap:
                print("[DEBUG CGEvent] Failed to create CGEventTap!")
                return False

            print("[DEBUG CGEvent] CGEventTap created successfully")

            # Create run loop source
            self.run_loop_source = Quartz.CFMachPortCreateRunLoopSource(
                None, self.tap, 0
            )

            # Add to current run loop
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetCurrent(),
                self.run_loop_source,
                Quartz.kCFRunLoopDefaultMode,
            )

            # Enable the tap
            Quartz.CGEventTapEnable(self.tap, True)

            print(
                "[DEBUG CGEvent] Hotkey listener started. Press Control+Option (⌃⌥) to record."
            )
            return True

    def stop(self):
        """Stop listening for hotkeys"""
        if self.running:
            if self.tap:
                Quartz.CGEventTapEnable(self.tap, False)
                if self.run_loop_source:
                    Quartz.CFRunLoopRemoveSource(
                        Quartz.CFRunLoopGetCurrent(),
                        self.run_loop_source,
                        Quartz.kCFRunLoopDefaultMode,
                    )
                self.tap = None
                self.run_loop_source = None

            self.running = False
            self.is_pressed = False
            # Shut down the dispatcher worker if we own it (only stop() does).
            # The sentinel is queued after any pending callbacks, so the
            # worker always terminates; a callback still running when stop()
            # is called finishes first. `start()` will not spawn a second
            # worker while the old one is still draining the shared queue
            # (is_alive guard in start()).
            if self._worker is not None:
                self._callback_queue.put(None)
                self._worker.join(timeout=5)
                self._worker = None
            print("[DEBUG CGEvent] Hotkey listener stopped.")

    def run_loop(self):
        """Run the event loop (blocks) - for CLI app"""
        try:
            Quartz.CFRunLoopRun()
        except KeyboardInterrupt:
            self.stop()

    def stop_loop(self):
        """Stop the event loop"""
        Quartz.CFRunLoopStop(Quartz.CFRunLoopGetCurrent())
