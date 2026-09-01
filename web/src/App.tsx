import { NavLink, Route, Routes } from "react-router-dom";
import { AccessibilityBar } from "./components/AccessibilityBar";
import { PlaceholderNotice } from "./components/PlaceholderNotice";
import { SignsProvider } from "./context/SignsContext";
import { Home } from "./pages/Home";
import { Learn } from "./pages/Learn";
import { Practice } from "./pages/Practice";
import { Progress } from "./pages/Progress";
import { Quiz } from "./pages/Quiz";
import { Record } from "./pages/Record";

function App() {
  return (
    <SignsProvider>
      <header className="app-header">
        <div className="app-title">Auslan Learning Assistant (prototype)</div>
        <nav className="app-nav">
          <NavLink to="/" end>
            Library
          </NavLink>
          <NavLink to="/quiz">Quiz</NavLink>
          <NavLink to="/practice">Practise</NavLink>
          <NavLink to="/progress">Progress</NavLink>
          <NavLink to="/record">Record (dev)</NavLink>
        </nav>
        <AccessibilityBar />
      </header>

      <PlaceholderNotice />

      <main className="app-main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/learn/:id" element={<Learn />} />
          <Route path="/quiz" element={<Quiz />} />
          <Route path="/practice" element={<Practice />} />
          <Route path="/practice/:id" element={<Practice />} />
          <Route path="/progress" element={<Progress />} />
          <Route path="/record" element={<Record />} />
        </Routes>
      </main>
    </SignsProvider>
  );
}

export default App;
