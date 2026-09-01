import { Route, Routes } from "react-router-dom";
import { SiteHeader } from "./components/SiteHeader";
import { HomePage } from "./pages/HomePage";
import { SignDetailPage } from "./pages/SignDetailPage";
import { SignLibraryPage } from "./pages/SignLibraryPage";

function App() {
  return (
    <>
      <SiteHeader />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/library" element={<SignLibraryPage />} />
        <Route path="/signs/:id" element={<SignDetailPage />} />
      </Routes>
    </>
  );
}

export default App;
