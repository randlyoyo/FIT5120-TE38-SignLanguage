import { Route, Routes } from "react-router-dom";
import { SignDetailPage } from "./pages/SignDetailPage";
import { SignLibraryPage } from "./pages/SignLibraryPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<SignLibraryPage />} />
      <Route path="/signs/:id" element={<SignDetailPage />} />
    </Routes>
  );
}

export default App;
