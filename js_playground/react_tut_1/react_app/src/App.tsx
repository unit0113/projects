import { useState } from "react";
import Alert from "./components/alert";
import Button from "./components/button";

function App() {
  const [alertVisible, setAlertVisibility] = useState(false);
  return (
    <div>
      {alertVisible && (
        <Alert onClose={() => setAlertVisibility(false)}>My Alert</Alert>
      )}
      <Button color="secondary" onClick={() => setAlertVisibility(true)}>
        Push Me!
      </Button>
    </div>
  );
}

export default App;
